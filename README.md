# Project KMAsgec

What is KMAsgec and why should you use it with your annotated species?

Often, we’ve wondered whether the annotations we have for a given species are truly correct—and that’s exactly why we designed KMAsgec. By capturing patterns through convolutional neural networks, KMAsgec attempts to predict how well an annotation has been performed.

The next question you might ask is: how was it designed to find these patterns? It’s simple: using k-mers and the annotation to be evaluated, we generate multiple tables of class frequencies to analyze (exon, intron, intergenic region, and transposable elements). These tables are produced using correctly annotated species for model training; as a result, our model has learned the patterns these probability‐distribution tables should follow, allowing us to predict the expected distribution and compare it against the one under evaluation.

What do these tables consist of? On the one hand, we have four columns representing the four classes we consider (exon, intron, intergenic region, and transposable elements); on the other hand, we have 4ᵏ rows, where k is the k-mer size. In our case, k is limited to values between 7 and 10, although a value of 7 is recommended. For each k-mer, these tables contain the probability of belonging to each of the four classes, and for a given genome, we generate multiple tables (the number depends on genome size).

Why multiple tables? It’s important that the model not see the overall, aggregated distribution, since our goal is for it to learn the distribution‐pattern features. Hence we don’t create just one table. Moreover, generating multiple tables gives us a more reliable assessment of annotation quality.

What does our model predict? For each input table, our model outputs a new table of identical dimensions; each cell in this output table contains the model’s final predicted probability for that specific k-mer and class.

How can we tell whether an annotation is good or bad? Our program reports the Kullback–Leibler divergence between the two distributions. Values between 0.30 and 0.50 suggest poor annotation quality, while values below 0.25 indicate a good annotation.

Which k-mers are the worst annotated? KMAsgec computes the Kullback–Leibler divergence for each k-mer between the two distributions; if this exceeds the 0.13 threshold, that k-mer is written to a file named **row_w_high_KLDivergence**, which you can review at the end of the program’s execution.

This project requires a Python virtual environment to manage dependencies in isolation. Below are the steps to create and activate the environment on different operating systems, and to install the package locally.

## Prerequisites

- Python 3.10 or higher
- `pip` (included with Python 3)
- Access to the terminal or command line

## 1. Create a virtual environment

From the project root folder, run:

```bash
python -m venv KMAsgec
```

### 1.1 Conda environment

Before installing the tool, you need to create another environment, this time with conda; so, from the terminal run:

```bash
conda create -n katulu \
  agat=1.2.0 \
  perl-clone \
  perl-list-moreutils \
  perl-sort-naturally \
  perl-try-tiny
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
pip install .
```

## 3. Run the program

Once the package is installed and the environment is activated, you can run the main program by passing the following arguments:

- **--gff**: Path to the GFF file.  
- **--fasta**: Path to the FASTA file.  
- **--n_cpus**: Number of CPUs to use.  
- **--repeatMask**: GFF3 file that enables annotation of transposable elements.  
- **--add_labels**: Adds introns, intergenic regions, and transposable elements using AGAT.  
- **--out**: Carpeta donde se va a alojar row_w_high_KLDivergence.


## Calculated metrics

- **Accuracy**: For each row, the highest probability is selected (e.g., column 2 – intron – with probability 0.8) and then checked against the evaluated annotation in the final distribution table to see if that row has the maximum probability in column 2.
- **Kullback–Leibler divergence**: Measures how much information is lost when approximating a probability distribution _P_ by another distribution _Q_.

