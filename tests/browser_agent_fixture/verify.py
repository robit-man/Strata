"""Independent browser acceptance test for the disposable agent exercise."""

import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("page", type=Path)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path="/usr/local/bin/chromium", headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page(viewport={"width": 900, "height": 700})
        try:
            # Flatpak Chromium sees a private /tmp, so load the disposable page's
            # content without assuming the browser shares the host's file paths.
            page.set_content(args.page.read_text(encoding="utf-8"))
            page.get_by_label("Observation").fill("Orion sensor online")
            page.get_by_role("button", name="Add entry").click()
            assert page.locator("#count").inner_text() == "1 entry"
            assert page.locator("#entries li").all_inner_texts() == ["Orion sensor online"]
            page.get_by_label("Filter entries").fill("missing")
            assert page.locator("#entries li").count() == 0
            page.get_by_label("Filter entries").fill("Orion")
            assert page.locator("#entries li").all_inner_texts() == ["Orion sensor online"]
            if args.screenshot:
                page.screenshot(path=str(args.screenshot), full_page=True)
            page.get_by_role("button", name="Clear all").click()
            assert page.locator("#count").inner_text() == "0 entries"
            print("PASS: add, count, filter, clear (real Chromium/Playwright)")
        finally:
            browser.close()


if __name__ == "__main__":
    main()
