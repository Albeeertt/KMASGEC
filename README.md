# Project KMAsgec

What is KMAsgec and why should you use it with your annotated species?

Often, we’ve wondered whether the annotations we have for a given species are truly correct—and that’s exactly why we designed KMAsgec. 
I am developing a Transformer encoder model to evaluate nucleotide sequences annotated in the GFF3 file provided as input, with the purpose of determining the correctness of each annotation.
Upon execution, a new GFF3 file is generated containing the same annotations, augmented with two additional columns specifying the probability assigned by the model and the predicted class for each annotation.
## Prerequisites

- Python 3.10 or higher
- `pip` (included with Python 3)
- Access to the terminal or command line

## 1. Create a virtual environment

From the project root folder, run:

```bash
git clone https://github.com/Albeeertt/KMASGEC.git
cd KMASGEC
python3.10 -m venv KMAsgec
```

### 1.1 Conda environment

Before installing the tool, you need to create another environment, this time with conda; so, from the terminal run:

```bash
conda config --set channel_priority strict
conda create -n katulu -c conda-forge -c bioconda agat=1.2.0 -y
```

### 1.2 Activate the virtual environment on Windows

From the project root folder, run:

```bash
KMAsgec\Scripts\activate
```

### 1.3 Activate the virtual environment on macOS and Linux

From the project root folder, run:

```bash
source KMAsgec/bin/activate
```


## 2. Access the project and install

From the terminal, navigate to the folder where this project is located (if you’re not already there) and install the package by running:

```bash
pip install . --no-cache-dir
```

## 3. Run the program

Once the package is installed and the environment is activated, you can run the main program by passing the following arguments:

- **--gff**: Path to the GFF file.  
- **--fasta**: Path to the FASTA file.   
- **--add_labels**: Adds introns, intergenic regions, and transposable elements using AGAT.  
- **--out**: Output directory where the newly generated GFF3 file will be stored.
- **--batch_size**: size of batch.
- **--gpus**: GPUs to be used, e.g., '0', '0,1', or '0,2,3'.
- **--small_algorithm**: The algorithm is switched to a smaller version with slightly reduced accuracy.
- **--train**: More aggressive model training to facilitate the learning of new patterns.
- **--fine_tunning**: Softer model training to enhance generalization across multiple species.

## 4. Example

```bash
KMAsgec --gff arabidopsis_thaliana.gff3 --fasta arabidopsis_thaliana.fasta --add_labels --out result_arabidopsis_thaliana --batch_size 12 --gpus 0,2
```



