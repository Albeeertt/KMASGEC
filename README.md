# Table of Contents 

- [What is KMASGEC?](#1-what-is-kmasgec)
- [Dependencies](#2-dependencies)
- [Items to install](#3-items-to-install)
- [Arguments](#4-arguments)
- [Example](#5-example)
- [Explanation of the output](#6-explanation-of-the-output)

# 1: What is KMASGEC?

What is KMAsgec and why should you use it with your annotated species?

Often, we’ve wondered whether the annotations we have for a given species are truly correct—and that’s exactly why we designed KMAsgec. 
I am developing a Transformer encoder model to evaluate nucleotide sequences annotated in the GFF3 file provided as input, with the purpose of determining the correctness of each annotation.
Upon execution, a new GFF3 file is generated containing the same annotations, augmented with two additional columns specifying the probability assigned by the model and the predicted class for each annotation.

# 2: Dependencies

You need to have Python version 3.10 and pip (included with Python 3).

Other library dependencies are handled by the *requeriments.txt* file and are resolved automatically.

# 3: Items to install


For the tool to work, copy and paste the following commands into the terminal depending on whether you are using macOS/Linux or Windows.

This terminal command sequence executes the following:

1. Downloads the program locally.
2. Creates a Python environment (version 3.10) to manage dependencies with other libraries.
3. Activates the environment.
4. Installs the program and the libraries specified in *requeriments.txt*

## 3.1: MacOS and Linux

```bash
git clone https://github.com/Albeeertt/KMASGEC.git
cd KMASGEC
python3.10 -m venv KMAsgec
git lfs install
git lfs pull

conda config --set channel_priority strict
conda create -n katulu -c conda-forge -c bioconda agat=1.2.0 -y

source KMAsgec/bin/activate

pip install . --no-cache-dir
```

## 3.2: Windows

```bash
git clone https://github.com/Albeeertt/KMASGEC.git
cd KMASGEC
python3.10 -m venv KMAsgec
git lfs install
git lfs pull

conda config --set channel_priority strict
conda create -n katulu -c conda-forge -c bioconda agat=1.2.0 -y

KMAsgec\Scripts\activate

pip install . --no-cache-dir
```


# 4: Arguments

Once the package is installed and the environment is activated, you can run the main program by passing the following arguments:

| Argument       | Explanation                          |
|-----------------|--------------------------------------|
| gff | Path to the GFF file. |
| fasta | Path to the FASTA file. |
| add_labels | Adds introns, intergenic regions, and transposable elements using AGAT. |
| out | Output directory where the newly generated GFF3 file will be stored. |
| batch_size | Size of batch. |
| gpus | GPUs to be used, e.g., '0', '0,1', or '0,2,3'. |
| max_len_seq | Maximum size of the nucleotide sequence that the model will handle. By default, this value is 10,000. |
| lens_mode | If this argument is added, sequences larger than the size assigned by the *max_len_seq* argument will be split. |
| zoom_len_seq | new value of *max_len_seq* for the sequences to be split| 

# 5: Example




```bash
KMAsgec --gff arabidopsis_thaliana.gff3 --fasta arabidopsis_thaliana.fasta --out result_arabidopsis_thaliana --add_labels --batch_size 16 --lens_mode --zoom_length 9999 --gpus 0,2
```

```bash
KMAsgec --gff arabidopsis_thaliana.gff3 --fasta arabidopsis_thaliana.fasta --out result_arabidopsis_thaliana --add_labels --batch_size 16 --lens_mode --zoom_length 9999 --gpus 0,2
```

# 5: Explanation of the output

