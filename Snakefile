configfile: "config.yaml"

import json

ENV = "environment.yaml"
TAG = config["output"]["tag"]
DIM = config["dimensions"]["dim"]
RUNS = range(200)

INIT_SAMPLES_DIR = f"saved_samples/{TAG}/"
INIT_SWD_DIR = f"saved_swd_distribution/{TAG}/"
SAMPLES_DIR = f"saved_samples/{TAG}/{DIM}D/"
SWD_DIR = f"saved_swd_distribution/{TAG}/{DIM}D/"
MODEL_DIR = f"saved_models/{TAG}/{DIM}D/"
WEIGHTS_DIR = f"saved_weights/{TAG}/{DIM}D/"
FIG_DIR = f"saved_figures/{TAG}/{DIM}D/"



rule initialize_analysis:
    input:
        NEUTfile=config["paths"]["neut_file"],
        GENIEfile=config["paths"]["genie_file"]
    
    params:
        mode=config["analysis"]["mode"],
        neutrino_PDG=config["analysis"]["neutrino_PDG"],
        train_percentage=config["analysis"]["train_percentage"],
        val_percentage=config["analysis"]["val_percentage"]
    
    output:
        samples_dir_3D=directory(INIT_SAMPLES_DIR + "3D/"),
        samples_dir_8D=directory(INIT_SAMPLES_DIR + "8D/"),
        last_sampled_file_3D = INIT_SAMPLES_DIR + "3D/target_test.csv",
        last_sampled_file_8D = INIT_SAMPLES_DIR + "8D/target_test.csv"

    conda:
        ENV
    
    shell:
        """
        python Init.py \
            --input_file_NEUT {input.NEUTfile} \
            --input_file_GENIE {input.GENIEfile} \
            --mode {params.mode} \
            --neutrino_PDG {params.neutrino_PDG} \
            --train_percentage {params.train_percentage} \
            --val_percentage {params.val_percentage} \
            --output_dir_samples_3D {output.samples_dir_3D} \
            --output_dir_samples_8D {output.samples_dir_8D} 
        """


rule run_initial_bootstrap_3D:
    input:
        samples_dir= INIT_SAMPLES_DIR + "3D/" + "target_test.csv",
        last_sampled_file = INIT_SAMPLES_DIR + "3D/target_test.csv"
    output:
        output_file=INIT_SWD_DIR + "3D/indiv_bootstrap/run_{run_id}.npy"
    conda:
        ENV
    shell:
        """
        python Bootstrap_swd.py \
            --distribution {input.samples_dir} \
            --output_dir {INIT_SWD_DIR}3D/indiv_bootstrap/ \
            --output_file {output.output_file} \
            --random_seed {wildcards.run_id}
        """

rule aggregate_bootstrap_3D:
    input:
        expand(INIT_SWD_DIR + "3D/indiv_bootstrap/run_{run_id}.npy", run_id=RUNS)
    output:
        final_file=INIT_SWD_DIR + "3D/swd_distribution_3D.npy"
    conda:
        ENV
    shell:
        """
        python - << 'EOF'
import numpy as np
arrays = [np.load(f) for f in {input}]
combined = np.concatenate(arrays, axis=0)
np.save("{output.final_file}", combined)
EOF
        """


rule run_initial_bootstrap_8D:
    input:
        samples_dir=INIT_SAMPLES_DIR + "8D/" + "target_test.csv",
        last_sampled_file = INIT_SAMPLES_DIR + "8D/target_test.csv"
    output:
        output_file=INIT_SWD_DIR + "8D/indiv_bootstrap/run_{run_id}.npy"
    conda:
        ENV
    shell:
        """
        python Bootstrap_swd.py \
            --distribution {input.samples_dir} \
            --output_dir {INIT_SWD_DIR}8D/indiv_bootstrap/ \
            --output_file {output.output_file} \
            --random_seed {wildcards.run_id}
        """

rule aggregate_bootstrap_8D:
    input:
        expand(INIT_SWD_DIR + "8D/indiv_bootstrap/run_{run_id}.npy", run_id=RUNS)
    output:
        final_file=INIT_SWD_DIR + "8D/swd_distribution_8D.npy"
    conda:
        ENV
    shell:
        """
        python - << 'EOF'
import numpy as np
arrays = [np.load(f) for f in {input}]
combined = np.concatenate(arrays, axis=0)
np.save("{output.final_file}", combined)
EOF
        """


rule custom_dim_analysis:
    input:
        NEUTfile=config["paths"]["neut_file"],
        GENIEfile=config["paths"]["genie_file"]
    
    params:
        mode=config["analysis"]["mode"],
        neutrino_PDG=config["analysis"]["neutrino_PDG"],
        train_percentage=config["analysis"]["train_percentage"],
        val_percentage=config["analysis"]["val_percentage"],
        parameters_interest=config["features"]["parameters_interest"],
        tag=config["output"]["tag"]
    
    output:
        samples_dir=directory(SAMPLES_DIR),
        last_sampled_file = SAMPLES_DIR + "target_test.csv"
    conda:
        ENV
    shell:
        """
        python Splitting-script.py \
            --input_file_NEUT {input.NEUTfile} \
            --input_file_GENIE {input.GENIEfile} \
            --mode {params.mode} \
            --neutrino_PDG {params.neutrino_PDG} \
            --train_percentage {params.train_percentage} \
            --val_percentage {params.val_percentage} \
            --parameters_interest {params.parameters_interest} \
            --samples_dir {output.samples_dir}
        """

rule run_custom_bootstrap:
    input:
        samples_dir= SAMPLES_DIR + "target_test.csv",
    output:
        output_file= SWD_DIR + "indiv_bootstrap/run_{run_id}.npy"
    conda:
        ENV
    shell:
        """
        python Bootstrap_swd.py \
            --distribution {input.samples_dir} \
            --output_dir {SWD_DIR}indiv_bootstrap/ \
            --output_file {output.output_file} \
            --random_seed {wildcards.run_id}
        """

rule aggregate_custom_bootstrap:
    input:
        expand(SWD_DIR + "indiv_bootstrap/run_{run_id}.npy", run_id=RUNS)
    output:
        final_file=SWD_DIR + f"swd_distribution_{DIM}D.npy"
    conda:
        ENV
    shell:
        """
        python - << 'EOF'
import numpy as np
arrays = [np.load(f) for f in {input}]
combined = np.concatenate(arrays, axis=0)
np.save("{output.final_file}", combined)
EOF
        """

    
rule train_models:
    input:
        samples_dir= SAMPLES_DIR,
        hparam_file=f"set_hyperparameters/{TAG}/hyperparameters.json",
        last_sampled_file = SAMPLES_DIR + "target_test.csv"
    
    params:
        model_list=config["models"]["model_list"],
        parameters_interest=config["features"]["parameters_interest"],
        model_list_str= config["models"]["model_list"]
    output:
        model_dir=directory(MODEL_DIR),
        weights_dir=directory(WEIGHTS_DIR),
        last_written_file = WEIGHTS_DIR + f"weights_path_dict_{DIM}D.json"
        
    conda:
        ENV
    shell:
        """
        python Training.py \
            --original_train {input.samples_dir}/original_train.csv \
            --original_val {input.samples_dir}/original_val.csv \
            --original_test {input.samples_dir}/original_test.csv \
            --target_train {input.samples_dir}/target_train.csv \
            --target_val {input.samples_dir}/target_val.csv \
            --target_test {input.samples_dir}/target_test.csv \
            --hparams_dict {input.hparam_file} \
            --save_weights_path {output.weights_dir} \
            --save_model_path {output.model_dir} \
            --model_list {params.model_list_str}
        """
        

rule compute_metrics_plots:
    input:
        original_test = INIT_SAMPLES_DIR + "8D/original_test.csv",
        target_test = INIT_SAMPLES_DIR + "8D/target_test.csv",
        weights_path = WEIGHTS_DIR + f"weights_path_dict_{DIM}D.json",
        swd_dist_file_3D = INIT_SWD_DIR + "3D/swd_distribution_3D.npy",
        swd_dist_file_8D = INIT_SWD_DIR + "8D/swd_distribution_8D.npy",
        swd_dist_file = SWD_DIR + f"swd_distribution_{DIM}D.npy"
        
    output:
        output_dir=directory(FIG_DIR),
        metrics_file=FIG_DIR + "metrics.json"
    
    params:
        parameters_interest=config["features"]["parameters_interest"]
    
    conda:
        ENV
    
    shell:
        """
        python Calc_Metrics.py \
            --original_test {input.original_test} \
            --target_test {input.target_test} \
            --weights_paths {input.weights_path} \
            --output_file {output.output_dir} \
            --make_1D_plots \
            --no-make_2D_plots \
            --compute_chi2 \
            --compute_swd \
            --interest_params {params.parameters_interest} \
            --custom_swd_distribution {input.swd_dist_file} \
            --swd_distribution_3D {input.swd_dist_file_3D} \
            --swd_distribution_8D {input.swd_dist_file_8D}
        """

rule all:
    input:
        FIG_DIR + "metrics.json"
    