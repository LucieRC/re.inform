"""
Scrape EVS variable metadata (variable code, question text, all value codes & labels)
for dataset ZA7505 (Joint EVS/WVS). Input: a list of variable short IDs like ["A001","A002",...].

Outputs:
 - pandas DataFrame with columns: dataset, var_short, var_full, question, value_code, value_label
 - saves CSV 'evs_variables_values.csv'

Notes:
 - First try: request JSON-LD from GESIS KG resource:
     https://data.gesis.org/gesiskg/resource/exploredata-ZA7505_Var{VAR}
   with Accept: application/ld+json
 - Fallback: use Selenium to render:
     https://search.gesis.org/variables/exploredata-ZA7505_Var{VAR}
 - Respect site policies (set a descriptive User-Agent, consider rate limits).
"""

import re
import json
import time
import requests
import pandas as pd
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup

# ----- User settings ----- #
DATASET = "ZA7505"
BASE_KG = "https://data.gesis.org/gesiskg/resource"  # HTML endpoint to find data links
BASE_SEARCH = "https://search.gesis.org/variables"
USER_AGENT = "evs-scraper/Lucie_Ricq)"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/ld+json"}
# ------------------------- #

# Example: variable_list = ["A001", "A002", "B010", ...]
def scrape_variables(variable_list: List[str]) -> pd.DataFrame:
    """
    Scrape GESIS variable pages using Firefox to render JavaScript content.
    
    Since GESIS search pages are JavaScript single-page applications, we go
    directly to Firefox rendering rather than attempting HTTP requests first.
    """
    rows = []
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    for var in variable_list:
        var = var.strip()
        print(f"[{var}] processing...")
        
        # GESIS pages are JavaScript SPAs - go straight to Firefox rendering
        selenium_result = scrape_with_selenium_for_variable(DATASET, var)
        
        if selenium_result:
            print(f"  SUCCESS: Extracted {var} - {selenium_result['question']}")
            if selenium_result.get("values"):
                print(f"  Found {len(selenium_result['values'])} value codes")
                for code, label in selenium_result["values"]:
                    rows.append({
                        "var_short": var,
                        "question": selenium_result.get("question"),
                        "value_code": code,
                        "value_label": label
                    })
            else:
                # Add single row without value codes
                rows.append({
                    "var_short": var,
                    "question": selenium_result.get("question"),
                    "value_code": None,
                    "value_label": None
                })
        else:
            print(f"  FAILED: Could not extract data for {var}")
            # Add fallback row
            rows.append({
                "var_short": var,
                "question": None,
                "value_code": None,
                "value_label": None
            })
        
        time.sleep(0.5)  # Be respectful to the server

    df = pd.DataFrame(rows, columns=["var_short", "question", "value_code", "value_label"])
    df.to_csv("variables_values_documentation.csv", index=False)
    print(f"\nResults saved to: variables_values_documentation.csv")
    return df

# ----------------------
# Helper: parse JSON-LD from KG
# ----------------------
def parse_jsonld_for_variable(j: Any) -> Optional[Dict[str, Any]]:
    """
    Very defensive JSON-LD parsing:
    - The structure may be a dict with '@graph' or a list.
    - We look for objects that contain 'variableName' or 'variableLabel' keys (or expanded URIs that end with variableName/variableLabel).
    - For code lists we look for nodes containing 'category' / 'hasCategory' / 'codeList' / 'skos:prefLabel' etc.
    """
    # normalize to list of nodes
    nodes = []
    if isinstance(j, dict):
        if "@graph" in j and isinstance(j["@graph"], list):
            nodes = j["@graph"]
        else:
            nodes = [j]
    elif isinstance(j, list):
        nodes = j
    else:
        return None

    # find the candidate variable node (heuristic)
    var_node = None
    for n in nodes:
        # keys can be expanded URIs or short keys; search any key's name for "variableName" or "variableLabel"
        key_text = " ".join(k.lower() for k in (n.keys() if isinstance(n, dict) else []))
        if "variablename" in key_text or "variablelabel" in key_text or "variable" in key_text:
            var_node = n
            break
    if not var_node:
        # fallback: find any node whose @type endswith 'Variable' or contains 'Variable'
        for n in nodes:
            t = n.get("@type") or n.get("type")
            if t:
                if isinstance(t, list):
                    ts = " ".join([str(x).lower() for x in t])
                else:
                    ts = str(t).lower()
                if "variable" in ts:
                    var_node = n
                    break
    if not var_node:
        return None

    out = {"var_full": None, "question": None, "values": []}

    # var_full: try id or label
    out["var_full"] = var_node.get("@id") or var_node.get("id") or var_node.get("variableName") or var_node.get("variableLabel")

    # question: try common properties
    for qkey in ["http://purl.org/dc/terms/description", "description", "http://schema.org/description", "questionText", "question_text", "question"]:
        q = var_node.get(qkey)
        if q:
            if isinstance(q, list): q = q[0]
            if isinstance(q, dict):
                # try extracting literal from {'@value': ...}
                q = q.get("@value") or q.get("value")
            out["question"] = _strip_html(str(q)).strip()
            break

    # look for value categories in node or in nodes referencing it
    # common keys: category, hasCategory, codeList, hasCodeList, 'http://rdf-vocabulary.ddialliance.org/category'
    values = []
    def _extract_values_from_node(n):
        local = []
        # try different possible fields
        candidate_keys = [k for k in n.keys()] if isinstance(n, dict) else []
        for k in candidate_keys:
            lk = k.lower()
            if any(x in lk for x in ["category", "codelist", "hascategory", "values", "code"]):
                v = n.get(k)
                if not v:
                    continue
                # v could be a list of nodes or a single node. Each node may be dict with prefLabel, rdfs:label, categoryLabel, or value
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            code, label = _extract_code_label_from_node(item)
                            if label is None and isinstance(item.get("@id"), str):
                                # maybe it's a reference to another node: try find that node in nodes
                                ref = item.get("@id")
                                ref_node = _find_node_by_id(nodes, ref)
                                if ref_node:
                                    code2, label2 = _extract_code_label_from_node(ref_node)
                                    if label2:
                                        local.append((code2, label2))
                            else:
                                if label is not None:
                                    local.append((code, label))
                elif isinstance(v, dict):
                    code, label = _extract_code_label_from_node(v)
                    if label is not None:
                        local.append((code, label))
                elif isinstance(v, str):
                    # string list e.g. "1: Yes; 2: No" try split
                    local += parse_inline_code_list(v)
        return local

    # try extract from variable node itself
    values += _extract_values_from_node(var_node)

    # try scanning all nodes for 'inScheme' or 'skos:ConceptScheme' references linked from var_node
    # look for nodes that appear to be a code list (type contains 'Category'/'CodeList'/'Skos')
    for n in nodes:
        t = n.get("@type") or n.get("type") or ""
        if isinstance(t, list):
            ts = " ".join(map(str, t)).lower()
        else:
            ts = str(t).lower() if t else ""
        if any(x in ts for x in ["category", "code", "skos:concept", "skos:conceptscheme", "concept"]):
            # is this node linked from var_node?
            # Heuristic: check if var_node references n (by @id) or vice versa
            nid = n.get("@id") or n.get("id")
            if not nid:
                continue
            # if var_node references it by value
            var_keys = var_node.keys() if isinstance(var_node, dict) else []
            linked = False
            for k in var_keys:
                v = var_node.get(k)
                if isinstance(v, list):
                    if any(isinstance(x, dict) and (x.get("@id")==nid or x.get("id")==nid) for x in v):
                        linked = True
                elif isinstance(v, dict):
                    if v.get("@id")==nid or v.get("id")==nid:
                        linked = True
            if linked:
                # extract categories from this node
                values += _extract_values_from_node(n)

    # if still empty, attempt to parse common textual fields that may contain inline code lists
    if not values:
        # search through textual fields in var_node
        for k, v in var_node.items():
            if isinstance(v, str):
                inline = parse_inline_code_list(v)
                if inline:
                    values += inline

    # deduplicate while preserving order
    seen = set()
    uniq = []
    for c,lbl in values:
        key = (str(c), str(lbl))
        if key not in seen:
            seen.add(key)
            uniq.append((c,lbl))
    out["values"] = uniq
    return out

# helpers used above
def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s)

def _find_node_by_id(nodes: List[Dict[str,Any]], nid: str) -> Optional[Dict[str,Any]]:
    for n in nodes:
        if n.get("@id")==nid or n.get("id")==nid:
            return n
    return None

def _extract_code_label_from_node(node: Dict[str,Any]):
    # try a variety of fields commonly used for code/label in JSON-LD
    label = None
    code = None
    for k in ["http://www.w3.org/2004/02/skos/core#prefLabel", "prefLabel", "rdfs:label", "label", "categoryLabel", "valueLabel", "skos:prefLabel"]:
        if k in node:
            v = node[k]
            if isinstance(v, list): v = v[0]
            if isinstance(v, dict):
                label = v.get("@value") or v.get("value") or None
            else:
                label = str(v)
            break
    for k in ["code", "value", "categoryCode", "http://www.w3.org/1999/02/22-rdf-syntax-ns#value"]:
        if k in node:
            code = node[k]
            if isinstance(code, dict):
                code = code.get("@value") or code.get("value")
            break
    # sometimes the node contains a single string like "1: Yes"
    if label is None and isinstance(node, str):
        parsed = parse_inline_code_list(node)
        if parsed:
            return parsed[0]
    return (code, label)

_inline_pair_re = re.compile(r"\s*([^\s:;\-]+)\s*[:\-]\s*(.+)")

def parse_inline_code_list(text: str):
    """
    Try to parse inline lists like:
      "1: Yes; 2: No; 8: Don't know"
    returns list of (code,label)
    """
    out = []
    if not isinstance(text, str):
        return out
    # split on semicolons or newline
    parts = re.split(r";|\n|\r", text)
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = _inline_pair_re.match(p)
        if m:
            out.append((m.group(1).strip(), _strip_html(m.group(2).strip())))
    return out

# ----------------------
# Helper: parse HTML content from GESIS search page
# ----------------------
def parse_html_for_variable(html_content: str, var_short: str) -> Optional[Dict[str, Any]]:
    """
    Parse HTML content from GESIS search page to extract variable metadata.
    
    Expected patterns in rendered JavaScript content:
    - Title: "GESIS-Suche: A001 - Important in life: Family"
    - Value table: <table class="variables_code_list_table">
    
    Returns None if the page appears to be unrendered JavaScript (SPA).
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract from title tag
        title_tag = soup.find('title')
        if not title_tag:
            print(f"  ERROR: No title tag found")
            return None
            
        title_text = title_tag.get_text()
        print(f"  Title: {title_text}")
        
        # Check if this looks like a JavaScript SPA that hasn't been rendered
        if ("GESIS-Suche" not in title_text and 
            "gesis" not in title_text.lower() and 
            var_short not in title_text):
            print(f"  WARNING: Title doesn't contain expected patterns")
            print(f"  This might be unrendered JavaScript content")
            
            # Additional checks for SPA indicators
            if any(indicator in html_content.lower() for indicator in [
                "angular", "react", "vue", "app-root", "ng-app", 
                "javascript required", "enable javascript"
            ]):
                print(f"  DETECTED: JavaScript framework indicators in HTML")
                print(f"  This page requires JavaScript rendering (Selenium needed)")
                return None
        
        # Parse title pattern: "GESIS-Suche: A001 - Important in life: Family"
        title_match = re.search(r'GESIS-Suche:\s*([A-Za-z0-9]+)\s*-\s*(.+)', title_text)
        if not title_match:
            print(f"  ERROR: Title doesn't match expected pattern")
            print(f"  Expected: 'GESIS-Suche: {var_short} - [Question Text]'")
            return None
            
        extracted_var = title_match.group(1)
        question_text = title_match.group(2)
        
        # Verify variable code matches
        if extracted_var != var_short:
            print(f"  WARNING: Variable mismatch. Expected {var_short}, got {extracted_var}")
        
        result = {
            "var_full": f"exploredata-{DATASET}_Var{var_short}",
            "question": question_text,
            "values": []
        }
        
        # Look for the value table: <table class="variables_code_list_table">
        value_table = soup.find('table', class_='variables_code_list_table')
        if value_table:
            print(f"  SUCCESS: Found variables_code_list_table")
            tbody = value_table.find('tbody')
            if tbody:
                rows = tbody.find_all('tr')
                print(f"  Found {len(rows)} table rows")
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        value_code = cells[0].get_text().strip()
                        value_label = cells[1].get_text().strip()
                        if value_code and value_label:
                            result["values"].append((value_code, value_label))
                            
                print(f"  SUCCESS: Extracted {len(result['values'])} value codes")
            else:
                print(f"  WARNING: Table found but no tbody element")
        else:
            print(f"  ERROR: No variables_code_list_table found")
            # Check if there are any table elements at all
            all_tables = soup.find_all('table')
            print(f"  Found {len(all_tables)} total table elements")
            
            # Check for common SPA loading indicators
            if any(indicator in html_content.lower() for indicator in [
                "loading", "please wait", "spinner", "loader"
            ]):
                print(f"  DETECTED: Loading indicators suggest JavaScript rendering needed")
                return None
        
        return result
        
    except Exception as e:
        print(f"  ERROR: HTML parsing exception: {e}")
        return None
def scrape_with_selenium_for_variable(dataset: str, var_short: str) -> Optional[Dict[str,Any]]:
    """
    Selenium scraper for JavaScript-rendered GESIS pages using Firefox.
    
    This function handles the dynamic content loading that happens after page load.
    It uses Firefox (which works well on macOS) to render the JavaScript content.
    
    Returns dict {var_full, question, values: [(code,label),...]} or None if failed.
    
    Prerequisites: pip install selenium webdriver-manager
    """
    try:
        # Lazy import so script doesn't force selenium if not needed
        from selenium import webdriver
        from selenium.webdriver.firefox.options import Options as FirefoxOptions
        from selenium.webdriver.firefox.service import Service as FirefoxService
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
        from webdriver_manager.firefox import GeckoDriverManager
    except ImportError as e:
        print(f"  ERROR: Selenium not available: {e}")
        print(f"  Install with: pip install selenium webdriver-manager")
        return None

    # Configure Firefox options
    options = FirefoxOptions()
    options.add_argument("--headless")
    options.set_preference("general.useragent.override", USER_AGENT)
    
    driver = None
    try:
        print(f"  Starting Firefox for JavaScript rendering...")
        service = FirefoxService(GeckoDriverManager().install())
        driver = webdriver.Firefox(service=service, options=options)
        
        url = f"https://search.gesis.org/variables/exploredata-{dataset}_Var{var_short}"
        driver.set_page_load_timeout(30)
        
        print(f"  Loading page: {url}")
        driver.get(url)
        
        # Wait for the page title to be updated (indicates JavaScript has run)
        try:
            WebDriverWait(driver, 15).until(
                lambda d: var_short in d.title or "GESIS-Suche:" in d.title
            )
            print(f"  Page rendered, title: {driver.title}")
        except TimeoutException:
            print(f"  WARNING: Page title didn't update within 15 seconds")
            print(f"  Current title: {driver.title}")
        
        # Additional wait for the content to load
        time.sleep(2)
        
        # Get the rendered HTML content
        html_content = driver.page_source
        print(f"  Captured {len(html_content)} characters of rendered HTML")
        
        # Use our existing HTML parser on the rendered content
        result = parse_html_for_variable(html_content, var_short)
        
        if result:
            print(f"  SUCCESS: Extracted variable data via Firefox")
        else:
            print(f"  ERROR: Could not parse rendered content")
            
        return result
        
    except Exception as e:
        print(f"  ERROR: Firefox automation failed: {e}")
        return None
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

# ----------------------
# Extract variable list from EVS data file
# ----------------------
def get_evs_variable_list(csv_path: str) -> List[str]:
    """
    Extract column names from EVS data file, starting from 'A001' onwards.
    
    Args:
        csv_path: Path to the EVS cleaned data CSV file
        
    Returns:
        List of variable codes (e.g., ['A001', 'A002', ...])
    """
    try:
        # Read just the header row to get column names
        df_header = pd.read_csv(csv_path, nrows=0)
        column_names = df_header.columns.tolist()
        
        print(f"Found {len(column_names)} total columns in EVS data")
        
        # Find the index of 'A001' column
        try:
            a001_index = column_names.index('A001')
            print(f"Found 'A001' at column index {a001_index}")
        except ValueError:
            print("ERROR: 'A001' column not found in the data")
            return []
        
        # Extract all columns from A001 onwards
        evs_variables = column_names[a001_index:]
        print(f"Extracted {len(evs_variables)} EVS variables starting from A001")
        print(f"First 10 variables: {evs_variables[:10]}")
        print(f"Last 10 variables: {evs_variables[-10:]}")
        
        return evs_variables
        
    except Exception as e:
        print(f"ERROR reading EVS data file: {e}")
        return []

# ----------------------
# Usage example
# ----------------------
if __name__ == "__main__":
    # Path to the EVS cleaned data file
    evs_data_path = "ZA7505_v5-0-0.dta/evs_cleaned_data.csv"
    
    # Extract variable list from the data file
    print("Extracting variable list from EVS data...")
    example_vars = get_evs_variable_list(evs_data_path)
    
    if not example_vars:
        print("Falling back to manual variable list...")
        example_vars = ["A001", "A002", "A003"]
    
    print(f"\nProcessing {len(example_vars)} variables...")
    
    # Uncomment the line below to process all variables (this will take a while!)
    # df = scrape_variables(example_vars)
    
    # For testing, let's just process the first few variables
    test_vars = example_vars[:5]  # First 5 variables for testing
    print(f"Testing with first {len(test_vars)} variables: {test_vars}")
    df = scrape_variables(test_vars)
    print(df.head())
