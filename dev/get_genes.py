from Bio import Entrez, SeqIO
import time
from dotenv import load_dotenv
import os
from io import StringIO  
import pandas as pd

# ENV init
load_dotenv()
Entrez.email = "jethrocsa@gmail.com"
Entrez.api_key = os.getenv('NCBI_KEY')

def gene_to_nuccore_ids(gene_name, organism=None, retmax=10):
    # Build term: gene name and optionally organism
    term = f"{gene_name}[Gene Name]"
    if organism:
        term += f" AND {organism}[Organism]"
    # search Gene DB
    handle = Entrez.esearch(db="gene", term=term, retmax=retmax)
    rec = Entrez.read(handle)
    handle.close()
    ids = rec.get("IdList", [])
    return ids

def geneid_to_nuccore(gene_id):
    # use elink to find linked nucleotide (nuccore) records
    h = Entrez.elink(dbfrom="gene", db="nuccore", id=gene_id)
    links = Entrez.read(h)
    h.close()
    # parse linked ids
    out_ids = []
    for linkset in links:
        for linksetdb in linkset.get("LinkSetDb", []):
            if linksetdb.get("DbTo") == "nuccore":
                out_ids.extend([l["Id"] for l in linksetdb.get("Link", [])])
    return out_ids

def fetch_gene_sequence(gid):
    # 1. Fetch Gene XML
    handle = Entrez.efetch(db="gene", id=gid, retmode="xml")
    records = Entrez.read(handle)
    handle.close()
    gene_info = records[0]
    locus = gene_info['Entrezgene_locus'][0]
    chrom_acc = locus['Gene-commentary_accession']
    start = int(locus['Gene-commentary_seqs'][0]['Seq-loc_int']['Seq-interval']['Seq-interval_from'])
    end   = int(locus['Gene-commentary_seqs'][0]['Seq-loc_int']['Seq-interval']['Seq-interval_to'])

    # 2. Fetch genomic sequence
    handle = Entrez.efetch(db="nuccore", id=chrom_acc, rettype="fasta", retmode="text", seq_start=start+1, seq_stop=end+1)
    fasta_text = handle.read()
    handle.close()
    time.sleep(0.34)  # polite delay
    seq_record = SeqIO.read(StringIO(fasta_text), "fasta")
    # 3. Reverse complement if minus strand
    return seq_record


# Example usage:
if __name__ == "__main__":

    # load datafiles
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, 'data')
    df_path = os.path.join(data_dir, 'toy_dataset_processed.csv')
    df = pd.read_csv(df_path)

    #parameters
    gene_list = pd.concat([df['target_gene'],df['reference_gene']])
    gene_list = gene_list.unique()
    organism = "Homo sapiens"
    gene_ids = []
    count = 0

    # get DMA seqiemces
    gene_record = []
    for gene_name in gene_list:
        gene_id = gene_to_nuccore_ids(gene_name, organism=organism, retmax=5)
        nuc_ids = geneid_to_nuccore(gene_id)
        record = fetch_gene_sequence(gene_id)
        time.sleep(0.34)

        # store
        row = {
            'gene': gene_name,
            'seq': str(record.seq)               
        }
        gene_record.append(row)


    # save to pdf
    save_df = pd.DataFrame(gene_record)
    save_path = os.path.join(data_dir,'gene_dna.csv')
    save_df.to_csv(save_path)
    print(f"Saved fle to {save_path}")



