conda create -y -n ldsc -c bioconda -c conda-forge python=2.7 bitarray=0.8 pybedtools=0.7 nose=1.3 pip
conda activate ldsc
wget https://bootstrap.pypa.io/pip/2.7/get-pip.py
python get-pip.py --force-reinstall
pip install scipy==0.18 pandas==0.20 numpy==1.16
git clone https://github.com/bulik/ldsc.git

### Download data (this could be moved to permanent folder to avoid redownloading everytime)
## Source: https://zenodo.org/records/10515792
wget https://zenodo.org/api/records/10515792/files/sumstats.tgz/content -O sumstats.tgz
wget https://zenodo.org/api/records/10515792/files/hm3_no_MHC.list.txt/content -O hm3_no_MHC.list.txt
wget https://zenodo.org/api/records/10515792/files/1000G_Phase3_plinkfiles.tgz/content -O 1000G_Phase3_plinkfiles.tgz
wget https://zenodo.org/api/records/10515792/files/1000G_Phase3_frq.tgz/content -O 1000G_Phase3_frq.tgz
wget https://zenodo.org/api/records/10515792/files/1000G_Phase3_weights_hm3_no_MHC.tgz/content -O 1000G_Phase3_weights_hm3_no_MHC.tgz
tar xvf 1000G_Phase3_plinkfiles.tgz
tar xvf 1000G_Phase3_weights_hm3_no_MHC.tgz
tar xvf 1000G_Phase3_frq.tgz
tar xvf sumstats.tgz

Rscript LDSC.R

for i in {1..22}; 
do python results/GWAS_enrich/ldsc/ldsc.py --l2 --bfile results/GWAS_enrich/1000G_EUR_Phase3_plink/1000G.EUR.QC.$i --ld-wind-cm 1 --annot results/GWAS_enrich/annotation/1000G.EUR.QC.$i.annot.gz --out results/GWAS_enrich/annotation/1000G.EUR.QC.$i --print-snps results/GWAS_enrich/hm3_no_MHC.list.txt;
done


seq 1 22 | xargs -n 1 -P 22 -I {} \
python results/GWAS_enrich/ldsc/ldsc.py \
    --l2 \
    --bfile results/GWAS_enrich/1000G_EUR_Phase3_plink/1000G.EUR.QC.{} \
    --ld-wind-cm 1 \
    --annot results/GWAS_enrich/annotation/1000G.EUR.QC.{}.annot.gz \
    --out results/GWAS_enrich/annotation/1000G.EUR.QC.{} \
    --print-snps results/GWAS_enrich/hm3_no_MHC.list.txt


python results/GWAS_enrich/ldsc/ldsc.py --h2 results/GWAS_enrich/sumstats/UKB_460K.body_LEFT_HANDED.sumstats.gz --ref-ld-chr results/GWAS_enrich/annotation/1000G.EUR.QC.,results/GWAS_enrich/baselineLD/baselineLD. --overlap-annot --frqfile-chr results/GWAS_enrich/1000G_Phase3_frq/1000G.EUR.QC. --out results/GWAS_enrich/results/Control_Handedness --w-ld-chr results/GWAS_enrich/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC. --print-coefficients

python results/GWAS_enrich/ldsc/ldsc.py --h2 results/GWAS_enrich/sumstats/UKB_460K.body_BMIz.sumstats.gz --ref-ld-chr results/GWAS_enrich/annotation/1000G.EUR.QC.,results/GWAS_enrich/baselineLD/baselineLD. --overlap-annot --frqfile-chr results/GWAS_enrich/1000G_Phase3_frq/1000G.EUR.QC. --out results/GWAS_enrich/results/Metabolic_BMI --w-ld-chr results/GWAS_enrich/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC. --print-coefficients

python results/GWAS_enrich/ldsc/ldsc.py --h2 results/GWAS_enrich/sumstats/UKB_460K.disease_T2D.sumstats.gz --ref-ld-chr results/GWAS_enrich/annotation/1000G.EUR.QC.,results/GWAS_enrich/baselineLD/baselineLD. --overlap-annot --frqfile-chr results/GWAS_enrich/1000G_Phase3_frq/1000G.EUR.QC. --out results/GWAS_enrich/results/Metabolic_T2D --w-ld-chr results/GWAS_enrich/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.  --print-coefficients


