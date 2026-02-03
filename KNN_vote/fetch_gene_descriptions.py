"""
fetch_gene_descriptions.py
==========================

Fetch descriptions for genes from databases including:
1. NCBI Gene Summary
2. UniProt (protein function)

Output:
- data/gene_name_to_ncbi.json
- data/gene_name_to_uniprot.json
- data/all_gene_descriptions.json (combined)
- data/all_gene_sources.json (tracks which source(s) each gene used)
"""

import re
import time
import requests
from xml.etree import ElementTree as ET
import mygene
import json
import os

mg = mygene.MyGeneInfo()

# Perturbations without GenePT embeddings (21)
GENE_LIST = ["AC118549.1", "ALG1L", "C12orf45", "C1orf109", "C7orf26",
            "C9orf16", "CCDC130", "CCDC144NL", "DDN", "FRG2", "MBTPS1",
            "PHB", "POLR2B", "SPATA5L1", "TDGF1", "WDR61", "WDR92", "ZNF720",
            "FAM102B", "INTS2", "SPATA5"]

# ============================================================================
# NCBI Scraper
# ============================================================================

def get_gene_summary_from_ncbi_api(gene_id):
    """
    Fetch gene summary  using NCBI's E-utilities API.
    
    Returns:
        summary text or empty string if not found
    """
    
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {'db': 'gene', 'id': gene_id, 'retmode': 'xml'}
    summary_text = ''
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        
        # Find Summary field
        summary_elem = root.find(".//Summary")
        if summary_elem is not None and summary_elem.text:
            summary_text = summary_elem.text.strip()
            if '[provided by' in summary_text:
                # Remove everything from "[provided by" onwards
                summary_text = summary_text.split('[provided by')[0]
            if '[supplied by' in summary_text:
                # Remove everything from "[supplied by" onwards
                summary_text = summary_text.split('[supplied by')[0]
        else:
            print("  NCBI: Summary tab not found")
    except Exception as e:
        print(f"  Error fetching gene {gene_id}: {e}")
        print(f"  NCBI: Failed with status {response.status_code}")
    
    return summary_text.strip()

# ============================================================================
# Uniprot scraper
# ============================================================================

def get_uniprot_description(gene_symbol):
    """
    Fetch protein function description from UniProt REST API.
    
    Returns:
        protein function text or empty string if not found
    """
    url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene_symbol}+AND+organism_id:9606&format=json&fields=protein_name,cc_function,gene_names"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get('results') and len(data['results']) > 0:
                result = data['results'][0]

                # Extract protein name
                protein_name = ""
                if 'proteinDescription' in result:
                    rec_name = result['proteinDescription'].get('recommendedName', {})
                    if 'fullName' in rec_name:
                        protein_name = rec_name['fullName'].get('value', '')
                
                # Extract function from comments
                function_text = ""
                if 'comments' in result:
                    for comment in result['comments']:
                        if comment.get('commentType') == 'FUNCTION':
                            texts = comment.get('texts', [])
                            if texts:
                                function_text = ' '.join([t.get('value', '') for t in texts])

                combined = f"{protein_name}. {function_text}".strip()
                return combined if combined != '.' else ''
    except Exception as e:
        print(f"  UniProt error: {e}")
    return ''

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":

    os.makedirs('data', exist_ok=True)

    # Step 1: Query mygene to get NCBI gene IDs
    print("Querying mygene for gene IDs...")
    query = mg.querymany(GENE_LIST, scopes='symbol', species='human')

    gene_name_to_entrez_id = {}
    for result in query:
        if "_id" in result and "query" in result:
            gene_name_to_entrez_id[result['symbol']] = result['_id']

    print(f"Found {len(gene_name_to_entrez_id)} genes in mygene: {list(gene_name_to_entrez_id.keys())}")
    genes_not_in_mygene = [g for g in GENE_LIST if g not in gene_name_to_entrez_id]
    print(f"Not found in mygene ({len(genes_not_in_mygene)}): {genes_not_in_mygene}")

    # Step 2: Fetch NCBI summaries for genes found in mygene
    print(f"\n{'='*60}")
    print("Fetching NCBI summaries...")
    print(f"{'='*60}")

    gene_name_to_ncbi = {}
    for gene_name, gene_id in sorted(gene_name_to_entrez_id.items()):
        print(f"\nProcessing: {gene_name} (ID: {gene_id})")
        desc = get_gene_summary_from_ncbi_api(gene_id)
        gene_name_to_ncbi[gene_name] = desc
        if desc:
            print(f"  Found NCBI summary ({len(desc)} chars)")
        else:
            print(f"  No NCBI summary")
        time.sleep(0.50)  # wait 0.50 seconds before next request

    # Save NCBI results
    with open('data/gene_name_to_ncbi.json', 'w') as f:
        json.dump(gene_name_to_ncbi, f, indent=2)
    print(f"\nSaved NCBI summaries to data/gene_name_to_ncbi.json")

    # Step 3: Fetch UniProt descriptions for ALL genes
    print(f"\n{'='*60}")
    print(f"Fetching from UniProt for all {len(GENE_LIST)} genes...")
    print(f"{'='*60}")

    gene_name_to_uniprot = {}

    for gene_name in sorted(GENE_LIST):
        print(f"\nProcessing: {gene_name}")
        print(f"  Trying UniProt...")
        desc = get_uniprot_description(gene_name)
        gene_name_to_uniprot[gene_name] = desc
        if desc:
            print(f"  Found in UniProt!")
        else:
            print(f"  No description found in UniProt")
        time.sleep(0.50)  # wait 0.50 seconds before next request

    # Save UniProt results
    with open('data/gene_name_to_uniprot.json', 'w') as f:
        json.dump(gene_name_to_uniprot, f, indent=2)
    print(f"\nSaved UniProt descriptions to data/gene_name_to_uniprot.json")

    # Step 4: Combine all descriptions
    print(f"\n{'='*60}")
    print("Combining all descriptions...")
    print(f"{'='*60}")

    all_descriptions = {}
    all_sources = {}  # Options: 'ncbi', 'uniprot', or 'ncbi+uniprot'

    for gene_name in GENE_LIST:
        ncbi_desc = gene_name_to_ncbi.get(gene_name, '').strip()
        uniprot_desc = gene_name_to_uniprot.get(gene_name, '').strip()

        if ncbi_desc and uniprot_desc:
            # Combine both descriptions
            all_descriptions[gene_name] = f"{uniprot_desc} {ncbi_desc}"
            all_sources[gene_name] = 'ncbi+uniprot'
        elif ncbi_desc:
            all_descriptions[gene_name] = ncbi_desc
            all_sources[gene_name] = 'ncbi'
        elif uniprot_desc:
            all_descriptions[gene_name] = uniprot_desc
            all_sources[gene_name] = 'uniprot'
        else:
            # Gene has no description from either source
            all_descriptions[gene_name] = ''
            all_sources[gene_name] = None

    # Save combined results
    with open('data/all_gene_descriptions.json', 'w') as f:
        json.dump(all_descriptions, f, indent=2)
    with open('data/all_gene_sources.json', 'w') as f:
        json.dump(all_sources, f, indent=2)

    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total genes in list: {len(GENE_LIST)}")

    genes_with_desc = [g for g in GENE_LIST if all_descriptions.get(g)]
    genes_without_desc = [g for g in GENE_LIST if not all_descriptions.get(g)]
    print(f"Genes with descriptions: {len(genes_with_desc)}")
    print(f"Genes without descriptions ({len(genes_without_desc)}): {genes_without_desc}")