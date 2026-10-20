// Portable Q2_0 CPU expert path for non-x86 hosts.  Native GGUF formats use
// ggml-cpu (including its AArch64 kernels); this scalar implementation keeps the
// original Q2_0 pack functional when no AVX implementation can be compiled.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/cpu/kq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, sizeof h);
    const uint32_t sign = uint32_t(h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; --exp; }
            bits = (sign << 31) | (exp << 23) | ((man & 0x3ffu) << 13);
        }
    } else if (exp == 31) {
        bits = (sign << 31) | 0x7f800000u | (man << 13);
    } else {
        bits = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof out);
    return out;
}

inline int q2(const uint8_t* codes, int j) {
    return int((codes[j >> 2] >> (2 * (j & 3))) & 3u) - 1;
}

float packed_row_dot(const uint8_t* codes, const uint8_t* scales, const ActQ& a, int nblocks) {
    float result = 0.0f;
    for (int b = 0; b < nblocks; ++b) {
        const float d = h2f(scales + 2 * b);
        for (int half = 0; half < 2; ++half) {
            const int chunk = 2 * b + half;
            int dot = 0;
            for (int j = 0; j < QKA; ++j)
                dot += q2(codes + 16 * b, half * QKA + j) * int(a.q[chunk * QKA + j]);
            result += d * a.scale[chunk] * float(dot);
        }
    }
    return result;
}

float gguf_row_dot(const uint8_t* row, const ActQ& a, int nblocks) {
    float result = 0.0f;
    for (int b = 0; b < nblocks; ++b) {
        const uint8_t* block = row + size_t(b) * BB;
        const float d = h2f(block);
        for (int half = 0; half < 2; ++half) {
            const int chunk = 2 * b + half;
            int dot = 0;
            for (int j = 0; j < QKA; ++j)
                dot += q2(block + 2, half * QKA + j) * int(a.q[chunk * QKA + j]);
            result += d * a.scale[chunk] * float(dot);
        }
    }
    return result;
}

void quantize(const float* x, int n, ActQ& a) {
    a.nchunks = n / QKA;
    a.bp_pairs = 0;
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + k * QKA;
        float amax = 0.0f;
        for (int j = 0; j < QKA; ++j) amax = std::fmax(amax, std::fabs(xb[j]));
        const float scale = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        int sum = 0;
        for (int j = 0; j < QKA; ++j) {
            const float v = xb[j] * inv;
            int q = int(v + (v >= 0.0f ? 0.5f : -0.5f));
            q = std::clamp(q, -127, 127);
            a.q[k * QKA + j] = int8_t(q);
            sum += q;
        }
        a.scale[k] = scale;
        a.sum[k] = sum;
        a.hx[k] = scale * float(sum);
    }
}

}  // namespace

void act_quant_q8_1(const float* x, int n, ActQ& a) { quantize(x, n, a); }
void act_quant_q8_1_avx2(const float* x, int n, ActQ& a) { quantize(x, n, a); }
void act_quant_q8_1_avx1(const float* x, int n, ActQ& a) { quantize(x, n, a); }
bool q2_bitplane_enabled() { return false; }

void bf16_rows_dot_multi(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    for (int r = 0; r < rows; ++r) {
        const uint16_t* wr = w + size_t(r) * size_t(cols);
        for (int t = 0; t < nt; ++t) {
            const float* xt = x + size_t(t) * size_t(cols);
            float sum = 0.0f;
            for (int c = 0; c < cols; ++c) sum += strata::kernels::f32_from_bf16(wr[c]) * xt[c];
            out[size_t(t) * size_t(rows) + r] = sum;
        }
    }
}

void bf16_rows_dot_multi_avx1(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    bf16_rows_dot_multi(w, rows, cols, x, nt, out);
}

void bf16_rows_dot(const uint16_t* w, int rows, int cols, const float* x, float* out) {
    bf16_rows_dot_multi(w, rows, cols, x, 1, out);
}

void s2_expert_gu_rows(const uint8_t* blob, const ActQ& a, float* ff, int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
        const float g = packed_row_dot(blob + O_GU_CODES + size_t(2 * r) * ROW_GU,
                                       blob + O_GU_SCALES + size_t(2 * r) * SC_GU * 2, a, SC_GU);
        const float u = packed_row_dot(blob + O_GU_CODES + size_t(2 * r + 1) * ROW_GU,
                                       blob + O_GU_SCALES + size_t(2 * r + 1) * SC_GU * 2, a, SC_GU);
        ff[r] = (g / (1.0f + std::exp(-g))) * u;
    }
}

void s2_expert_down_rows(const uint8_t* blob, const ActQ& a, float* out, int r0, int r1) {
    for (int r = r0; r < r1; ++r)
        out[r] = packed_row_dot(blob + O_D_CODES + size_t(r) * ROW_D,
                                blob + O_D_SCALES + size_t(r) * SC_D * 2, a, SC_D);
}

void s2_expert_vnni_q(const uint8_t* blob, const ActQ& a, float* out, ExpertScratch& ws) {
    s2_expert_gu_rows(blob, a, ws.ff, 0, FF);
    quantize(ws.ff, FF, ws.a2);
    s2_expert_down_rows(blob, ws.a2, out, 0, H);
}

void s2_expert_vnni(const uint8_t* blob, const float* x, float* out, ExpertScratch& ws) {
    quantize(x, H, ws.a1);
    s2_expert_vnni_q(blob, ws.a1, out, ws);
}

void s2_expert_gu_rows_multi(const uint8_t* blob, const ActQ* const* act, int nt, float* const* ff, int r0, int r1) {
    for (int t = 0; t < nt; ++t) s2_expert_gu_rows(blob, *act[t], ff[t], r0, r1);
}

void s2_expert_down_rows_multi(const uint8_t* blob, const ActQ* const* act, int nt, float* const* out,
                               int r0, int r1) {
    for (int t = 0; t < nt; ++t) s2_expert_down_rows(blob, *act[t], out[t], r0, r1);
}

void s2_expert_vnni_multi(const uint8_t* blob, const ActQ* const* act, int nt, float* const* out,
                          ExpertScratchMulti& ws) {
    for (int t = 0; t < nt; ++t) s2_expert_vnni_q(blob, *act[t], out[t], ws.single);
}

void q2_0_gguf_rows_multi(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* act, int nt,
                          float* const* out, int r0, int r1) {
    for (int r = r0; r < r1; ++r)
        for (int t = 0; t < nt; ++t)
            out[t][r] = gguf_row_dot(w + size_t(r) * row_bytes, *act[t], nblocks);
}

void q2_0_gguf_rows_multi_avx2(const uint8_t* w, size_t rb, int nb, const ActQ* const* a, int nt,
                               float* const* out, int r0, int r1) {
    q2_0_gguf_rows_multi(w, rb, nb, a, nt, out, r0, r1);
}
void q2_0_gguf_rows_multi_avx2_legacy(const uint8_t* w, size_t rb, int nb, const ActQ* const* a, int nt,
                                      float* const* out, int r0, int r1) {
    q2_0_gguf_rows_multi(w, rb, nb, a, nt, out, r0, r1);
}
void q2_0_gguf_rows_multi_avx2_v(bool, const uint8_t* w, size_t rb, int nb, const ActQ* const* a, int nt,
                                 float* const* out, int r0, int r1) {
    q2_0_gguf_rows_multi(w, rb, nb, a, nt, out, r0, r1);
}
void q2_0_gguf_rows_multi_avx1(const uint8_t* w, size_t rb, int nb, const ActQ* const* a, int nt,
                               float* const* out, int r0, int r1) {
    q2_0_gguf_rows_multi(w, rb, nb, a, nt, out, r0, r1);
}
void q2_0_gguf_rows_multi_avx1_legacy(const uint8_t* w, size_t rb, int nb, const ActQ* const* a, int nt,
                                      float* const* out, int r0, int r1) {
    q2_0_gguf_rows_multi(w, rb, nb, a, nt, out, r0, r1);
}

void s2_expert_scalar(const uint8_t* blob, const float* x, float* out, bool quant_acts) {
    if (quant_acts) {
        ExpertScratch ws;
        s2_expert_vnni(blob, x, out, ws);
        return;
    }
    float ff[FF];
    for (int r = 0; r < FF; ++r) {
        float g = 0.0f, u = 0.0f;
        for (int b = 0; b < SC_GU; ++b) {
            const float dg = h2f(blob + O_GU_SCALES + size_t(2 * r) * SC_GU * 2 + 2 * b);
            const float du = h2f(blob + O_GU_SCALES + size_t(2 * r + 1) * SC_GU * 2 + 2 * b);
            const uint8_t* gc = blob + O_GU_CODES + size_t(2 * r) * ROW_GU + 16 * b;
            const uint8_t* uc = blob + O_GU_CODES + size_t(2 * r + 1) * ROW_GU + 16 * b;
            for (int j = 0; j < QK; ++j) {
                g += float(q2(gc, j)) * dg * x[b * QK + j];
                u += float(q2(uc, j)) * du * x[b * QK + j];
            }
        }
        ff[r] = (g / (1.0f + std::exp(-g))) * u;
    }
    for (int r = 0; r < H; ++r) {
        float sum = 0.0f;
        for (int b = 0; b < SC_D; ++b) {
            const float d = h2f(blob + O_D_SCALES + size_t(r) * SC_D * 2 + 2 * b);
            const uint8_t* c = blob + O_D_CODES + size_t(r) * ROW_D + 16 * b;
            for (int j = 0; j < QK; ++j) sum += float(q2(c, j)) * d * ff[b * QK + j];
        }
        out[r] = sum;
    }
}

}  // namespace strata::kernels::cpu
