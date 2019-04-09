import argparse
import os
import shutil

from toil.common import Toil
from toil.job import Job
from toil.lib.docker import apiDockerCall, _fixPermissions


def workflow(job, samples, args):
    sample_id = job.fileStore.writeGlobalFile(args.sample)
    background_id = job.fileStore.writeGlobalFile(args.background)
    gene_id = job.fileStore.writeGlobalFile(args.gene_list) if args.gene_list else None

    job.addChildJobFn(
        map_job, run_outlier_model, samples, sample_id, background_id, gene_id, args
    )


def run_outlier_model(
    job, name, sample_id, background_id, gene_id, args, cores=2, memory="5G"
):
    # Check if output already exists and don't run if so
    output = os.path.join(args.out_dir, name)
    if os.path.exists(output):
        return 0

    # Process names with flexible extensions
    sample_ext = os.path.splitext(args.sample)[1]
    sample_name = "sample_matrix{}".format(sample_ext)
    bg_ext = os.path.splitext(args.background)[1]
    bg_name = "bg_matrix{}".format(bg_ext)

    # Read in input file from jobStore
    job.fileStore.readGlobalFile(sample_id, os.path.join(job.tempDir, sample_name))
    job.fileStore.readGlobalFile(background_id, os.path.join(job.tempDir, bg_name))
    if gene_id:
        job.fileStore.readGlobalFile(
            gene_id, os.path.join(job.tempDir, "gene-list.txt")
        )

    # Define parameters and call Docker container
    parameters = [
        "--sample",
        "/data/{}".format(sample_name),
        "--background",
        "/data/{}".format(bg_name),
        "--name",
        name,
        "--out-dir",
        "/data",
        "--group",
        args.group,
        "--col-skip",
        str(args.col_skip),
        "--num-backgrounds",
        str(args.num_backgrounds),
        "--max-genes",
        str(args.max_genes),
        "--num-training-genes",
        str(args.num_training_genes),
        "--pval-convergence-cutoff",
        str(args.pval_convergence_cutoff),
    ]
    if args.disable_iter:
        parameters.append("--disable-iter")
    if gene_id:
        parameters.extend(["--gene-list", "/data/gene-list.txt"])
    image = "jvivian/gene-outlier-detection"
    apiDockerCall(
        job=job,
        image=image,
        working_dir=job.tempDir,
        parameters=parameters,
        user="root",
    )
    _fixPermissions(tool=image, workDir=job.tempDir)

    out_dir = os.path.join(job.tempDir, name)
    shutil.move(out_dir, args.out_dir)


def cli():
    parser = argparse.ArgumentParser(
        description=main.__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--sample",
        required=True,
        type=str,
        help="Sample(s) by Genes matrix (csv/tsv/hd5)",
    )
    parser.add_argument(
        "--background",
        required=True,
        type=str,
        help="Samples by Genes matrix with metadata columns first (including a group column that "
        "discriminates samples by some category) (csv/tsv/hd5)",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=str,
        help="Single column file of sample names in sample matrix",
    )
    parser.add_argument(
        "--gene-list", type=str, help="Single column file of genes to use for training"
    )
    parser.add_argument("--out-dir", default=".", type=str, help="Output directory")
    parser.add_argument(
        "--group",
        default="tissue",
        type=str,
        help="Categorical column vector in the background matrix",
    )
    parser.add_argument(
        "--col-skip",
        default=1,
        type=int,
        help="Number of metadata columns to skip in background matrix so remainder are genes",
    )
    parser.add_argument(
        "--num-backgrounds",
        default=5,
        type=int,
        help="Number of background categorical groups to include in the model training",
    )
    parser.add_argument(
        "--max-genes",
        default=100,
        type=int,
        help="Maximum number of genes to run. I.e. if a gene list is input, how many additional"
        "genes to add via SelectKBest. Useful for improving beta coefficients"
        "if gene list does not contain enough tissue-specific genes.",
    )
    parser.add_argument(
        "--num-training-genes",
        default=50,
        type=int,
        help="If gene-list is empty, will use SelectKBest to choose gene set.",
    )
    parser.add_argument(
        "--pval-convergence-cutoff",
        default=0.99,
        type=float,
        help="P-value Pearson correlation cutoff to stop adding additional background datasets.",
    )
    parser.add_argument(
        "--disable-iter",
        action="store_true",
        help="This flag disables iterative runs and runs one model with `--num-backgrounds`",
    )

    # Add Toil options
    Job.Runner.addToilOptions(parser)
    return parser.parse_args()


def map_job(job, func, inputs, *args, **kwargs):
    """
    Spawns a tree of jobs to avoid overloading the number of jobs spawned by a single parent.
    This function is appropriate to use when batching samples greater than 1,000.

    :param JobFunctionWrappingJob job: passed automatically by Toil
    :param function func: Function to spawn dynamically, passes one sample as first argument
    :param list inputs: Array of samples to be batched
    :param list args: any arguments to be passed to the function
    """
    # num_partitions isn't exposed as an argument in order to be transparent to the user.
    # The value for num_partitions is a tested value
    num_partitions = 100
    partition_size = len(inputs) / num_partitions
    if partition_size > 1:
        for partition in partitions(inputs, partition_size):
            job.addChildJobFn(map_job, func, partition, *args, **kwargs)
    else:
        for sample in inputs:
            job.addChildJobFn(func, sample, *args, **kwargs)


def partitions(l, partition_size):
    """
    >>> list(partitions([], 10))
    []
    >>> list(partitions([1,2,3,4,5], 1))
    [[1], [2], [3], [4], [5]]
    >>> list(partitions([1,2,3,4,5], 2))
    [[1, 2], [3, 4], [5]]
    >>> list(partitions([1,2,3,4,5], 5))
    [[1, 2, 3, 4, 5]]

    :param list l: List to be partitioned
    :param int partition_size: Size of partitions
    """
    for i in xrange(0, len(l), partition_size):
        yield l[i : i + partition_size]


def main():
    args = cli()
    samples = [
        x.strip() for x in open(args.manifest, "r").readlines() if not x.isspace()
    ]

    # Start Toil run
    with Toil(args) as toil:
        if not toil.options.restart:
            toil.start(Job.wrapJobFn(workflow, samples, args))
        else:
            toil.restart()


if __name__ == "__main__":
    main()
