import os
import asyncio
from typing import Optional
from playwright.async_api import async_playwright
import config

class PlaywrightYouTubeService:
    """Service for downloading YouTube videos using Playwright"""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        
    async def download_video(self, youtube_url: str, output_path: str) -> str:
        """
        Download YouTube video using vidssave.com via Playwright
        
        Args:
            youtube_url (str): YouTube video URL
            output_path (str): Full path where to save the video
        
        Returns:
            str: Path to downloaded file
        """
        print(f"Starting Playwright download for: {youtube_url}")
        print(f"Target path: {output_path}")
        
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        async with async_playwright() as p:
            # Launch browser
            # Note: We must ensure chromium is installed (playwright install chromium)
            browser = await p.chromium.launch(headless=self.headless)
            context = await browser.new_context(
                accept_downloads=True,
                viewport={'width': 1280, 'height': 800},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            
            # Set default timeout
            context.set_default_timeout(60000)  # 60 seconds
            
            page = await context.new_page()
            
            try:
                # Navigate to vidssave.com
                print("Navigating to vidssave.com...")
                await page.goto("https://vidssave.com/yt", wait_until="domcontentloaded")
                
                # Handle possible cookie/popup
                try:
                    cookie_buttons = await page.query_selector_all('button:has-text("Accept"), button:has-text("OK"), button:has-text("Agree")')
                    if cookie_buttons:
                        await cookie_buttons[0].click()
                        await page.wait_for_timeout(1000)
                except:
                    pass
                
                # Find and fill the YouTube URL input
                print("Entering YouTube URL...")
                url_input = await page.wait_for_selector('input[placeholder*="YouTube"], input[type="url"], input[name*="url"]', state="visible", timeout=15000)
                await url_input.fill(youtube_url)
                
                # Click the download/submit button
                print("Submitting URL...")
                submit_button = await page.wait_for_selector('button:has-text("Download"), button:has-text("Submit"), button[type="submit"]', state="visible", timeout=5000)
                await submit_button.click()
                
                # Wait for processing
                print("Waiting for video options...")
                # Increase wait time for processing results
                await page.wait_for_timeout(5000) 
                
                # Selectors for quality
                quality_selectors = [
                    'a:has-text("720"), a:has-text("1080"), a:has-text("MP4")',
                    'button:has-text("720"), button:has-text("1080")',
                    '.download-link',
                    '.quality-option'
                ]
                
                download_link = None
                
                # Try finding best quality
                for selector in quality_selectors:
                    try:
                        elements = await page.query_selector_all(selector)
                        for element in elements:
                            text = await element.text_content() or ""
                            # Prioritize MP4 and HD
                            if any(qual in text for qual in ["720", "1080", "MP4", "Download"]):
                                download_link = element
                                break
                        if download_link:
                            break
                    except:
                        continue
                
                if not download_link:
                    # Fallback generic download link
                    download_link = await page.query_selector('a[href*="download"], button:has-text("Download")')
                
                if download_link:
                    print(f"Found download link: {await download_link.text_content()}")
                    print("Initiating download...")
                    
                    async with page.expect_download(timeout=120000) as download_info:
                        # Some sites open a new tab on click, or simple download
                        await download_link.click()
                        
                        download = await download_info.value
                        
                        # Save to target path
                        await download.save_as(output_path)
                        
                        print(f"Download complete: {output_path}")
                        
                        # Verify file
                        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                            return output_path
                        else:
                            raise Exception("File saved but appears empty")
                else:
                    # Take screenshot for debugging
                    screenshot_path = os.path.join(output_dir, "error_screenshot.png")
                    await page.screenshot(path=screenshot_path)
                    print(f"Could not find download link. Screenshot saved to {screenshot_path}")
                    raise Exception("Could not find download link on page")
                    
            except Exception as e:
                print(f"Error during Playwright download: {str(e)}")
                try:
                    screenshot_path = os.path.join(output_dir, "error_screenshot.png")
                    await page.screenshot(path=screenshot_path)
                    print(f"Error screenshot saved to: {screenshot_path}")
                except:
                    pass
                raise e
                
            finally:
                await browser.close()

# Singleton instance
playwright_youtube_service = PlaywrightYouTubeService()
