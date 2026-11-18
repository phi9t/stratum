"""Behavioral checks against the actual built subpath, using the parent uv Playwright."""
import functools
import http.server
import threading
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        return super().translate_path('/' + path.removeprefix('/parallel-training/'))
    def log_message(self, *args):
        pass

def main():
    with http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=str(ROOT/'dist'))) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f'http://127.0.0.1:{server.server_port}/parallel-training/'
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={'width':1440,'height':1000}, reduced_motion='reduce')
            page.set_default_timeout(10000)
            errors, external = [], []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('request', lambda request: external.append(request.url) if not request.url.startswith(base) else None)
            page.goto(base)
            page.get_by_role('heading',name='Model dependencies').wait_for()
            page.locator('.semantic-flow .react-flow__node').first.wait_for()
            assert page.locator('.inventory').count() == 4
            assert page.locator('.messages .message-table tbody tr').count() == 6
            page.get_by_label('Operation',exact=True).select_option('dw-up')
            page.locator('.module-controls').get_by_role('button',name='Collapse Backward',exact=True).click()
            page.get_by_text('selected inside collapsed module',exact=False).wait_for()
            assert page.get_by_label('Operation',exact=True).input_value() == 'dw-up'
            page.get_by_role('button',name='Evidence & execution',exact=True).click()
            assert page.get_by_text('0 observed runtime events',exact=True).is_visible()
            page.go_back()
            page.locator('.module-controls').get_by_role('button',name='Expand Backward',exact=True).wait_for()
            page.go_back()
            page.locator('.module-controls').get_by_role('button',name='Collapse Backward',exact=True).wait_for()
            assert page.get_by_label('Operation',exact=True).input_value() == 'dw-up'
            page.get_by_label('Weights after forward',exact=True).select_option('retain')
            assert page.get_by_label('Operation',exact=True).locator('option[value=gather-b]').count() == 0
            page.get_by_label('Operation',exact=True).select_option('dx')
            assert page.locator('.rank-tp').count() == 8
            page.get_by_role('button',name='FSDP column 0',exact=True).click()
            assert page.locator('.rank-fsdp').count() == 2
            page.get_by_role('button',name='Rank 9',exact=True).click()
            assert page.get_by_role('button',name='FSDP column 1',exact=True).is_visible()
            page.get_by_label('Graph',exact=True).select_option('attention')
            page.get_by_label('Operation',exact=True).select_option('add')
            assert page.locator('.graph-inspector').get_by_text('Residual add',exact=True).is_visible()
            page.locator('.module-controls').get_by_role('button',name='Collapse Attention',exact=True).click()
            page.locator('.semantic-flow .react-flow__node').first.wait_for()
            assert page.locator('.rank-mesh').count() == 0
            page.get_by_label('Graph',exact=True).select_option('sparse')
            assert page.get_by_text('Illustrative Civet-style lookup/mixer motif.',exact=False).is_visible()
            page.get_by_label('Operation',exact=True).select_option('lookup')
            page.get_by_role('button',name='Evidence & execution',exact=True).click()
            assert page.get_by_text('0 observed runtime events',exact=True).is_visible()
            page.reload()
            assert page.get_by_label('Graph',exact=True).input_value() == 'sparse'
            assert page.get_by_label('Operation',exact=True).input_value() == 'lookup'
            assert page.get_by_role('button',name='Evidence & execution',exact=True).get_attribute('aria-pressed') == 'true'
            page.get_by_role('button',name='Dependencies',exact=True).click()
            page.get_by_label('Graph',exact=True).select_option('mlp')
            page.locator('.scenario-controls select').first.select_option('2')
            assert page.locator('.messages .message-table tbody tr').count() == 14
            page.locator('.stage-head').filter(has_text='Stage 2 · C2').click()
            assert page.get_by_label('Layer',exact=True).input_value() == '29'
            page.get_by_label('Operation',exact=True).select_option('dw-down')
            page.locator('.module-controls').get_by_role('button',name='Collapse Forward',exact=True).click()
            page.locator('.semantic-flow .react-flow__node').first.wait_for()
            page.screenshot(path=str(ROOT/'tests'/'graph-desktop.png'), full_page=True)
            # A reusable graph can occupy a narrow pane even at desktop width.
            page.locator('#mlp').evaluate("e => e.style.width='360px'")
            assert page.locator('#mlp').evaluate('e => e.scrollWidth <= e.clientWidth')
            page.locator('#mlp').evaluate("e => e.style.removeProperty('width')")
            page.set_viewport_size({'width':320,'height':850})
            assert page.evaluate('document.documentElement.scrollWidth === innerWidth')
            fixtures=page.get_by_label('Graph',exact=True).locator('option').evaluate_all('els => els.map(e => e.value)')
            for graph in fixtures:
                page.get_by_label('Graph',exact=True).select_option(graph)
                page.locator('.semantic-flow .react-flow__node').first.wait_for()
                assert page.evaluate('document.documentElement.scrollWidth === innerWidth')
                clipped = page.locator('.semantic-flow .mlp-node button').evaluate_all(
                    'els => els.filter(e => e.scrollHeight > e.clientHeight || e.scrollWidth > e.clientWidth).length')
                assert clipped == 0
                page.locator('.graph-tabs button').first.focus()
                page.keyboard.press('Tab')
                assert page.locator('.graph-tabs button').nth(1).evaluate('e => e.matches(":focus-visible")')
                # Every native control has a name and remains in the keyboard order.
                unnamed = page.locator('button,select').evaluate_all("""els => els.filter(e => !e.getAttribute('aria-label') && !e.textContent.trim() && !e.labels?.length).length""")
                assert unnamed == 0
            page.screenshot(path=str(ROOT/'tests'/'graph-mobile.png'), full_page=True)
            assert not errors, errors
            assert not external, external
            print(f'PASS Chromium {browser.version}: all fixtures, collapse/selection, Back/Forward, ranks, evidence, schedules, 320px, offline assets')
            import json
            report = {'status':'PASS','browser':'Chromium '+browser.version,
                'fixtures':fixtures,
                'checks':['collapse/selection','Back/Forward','rank groups','evidence scope','1F1B regression',
                          '320px overflow','node text clearance','keyboard focus','accessible controls','offline assets']}
            (ROOT/'tests'/'browser-results.json').write_text(json.dumps(report,indent=2)+'\n')
            browser.close()

if __name__ == '__main__':
    main()
