"""
Generates a dataset for training on a SLURM cluster.
Each route file is parallelized on its own machine.
Monitors the data collection and continues crashed processes.
Best run inside a tmux terminal.
"""

import glob
import subprocess
import time
from pathlib import Path
import os
import sys
import json
import fnmatch


def make_jobsub_file(commands, job_number, exp_name, dataset_name, root_folder, node, mail):
  os.makedirs(f'{root_folder}{dataset_name}_logs/run_files/logs', exist_ok=True)
  os.makedirs(f'{root_folder}{dataset_name}_logs/run_files/job_files', exist_ok=True)
  job_file = f'{root_folder}{dataset_name}_logs/run_files/job_files/{job_number}.sh'
  qsub_template = f"""#!/bin/bash
#SBATCH --job-name={exp_name}{job_number}
#SBATCH --partition={node}
#SBATCH --account=share-ie-idi
#SBATCH -o {root_folder}{dataset_name}_logs/run_files/logs/qsub_out{job_number}.log
#SBATCH -e {root_folder}{dataset_name}_logs/run_files/logs/qsub_err{job_number}.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user={mail}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16gb
#SBATCH --cpus-per-task=10
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=(a100|v100)

# print info about current job
scontrol show job $SLURM_JOB_ID 

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "Job started: $dt"

echo "Current branch:"
git branch
echo "Current commit:"
git log -1
echo "Current hash:"
git rev-parse HEAD

nvidia-smi

"""
  for cmd in commands:
    qsub_template = qsub_template + f"""
{cmd}

"""

  with open(job_file, 'w', encoding='utf-8') as f:
    f.write(qsub_template)
  return job_file


def get_num_jobs(job_name, username, code_root):
  num_running_jobs = int(
      subprocess.check_output(
          f'SQUEUE_FORMAT2=\'username:{len(username)},name:130\' squeue --sort V | '
          f'grep {username} | grep {job_name} | wc -l',
          shell=True,
      ).decode('utf-8').replace('\n', ''))
  with open(f'{code_root}/max_num_jobs.txt', 'r', encoding='utf-8') as f:
    max_num_parallel_jobs = int(f.read())

  return num_running_jobs, max_num_parallel_jobs


def get_carla_command(gpu, num_try, start_port, carla_root):
  command = f'sh {carla_root}/CarlaUE4.sh ' \
          f'--world-port=$FREE_WORLD_PORT -RenderOffScreen -nosound -graphicsadapter=0 -carla-streaming-port=0'
  return command


def create_run_eval_bash(bash_save_dir, route_path, scenario_path, gpu, iteration, route, scen, repetition, dataset,
                         checkpoint, agent, start_port, carla_root, code_root, lib_path):
  with open(f'{bash_save_dir}/run_autopilot_{route}_Repetition{repetition}.sh', 'w', encoding='utf-8') as rsh:
    rsh.write(f'''\
#! /bin/bash
export CARLA_ROOT={carla_root}
export CARLA_SERVER=${{CARLA_ROOT}}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${{CARLA_ROOT}}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${{CARLA_ROOT}}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${{CARLA_ROOT}}/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=/cluster/work/andrebw/carla_garage/scenario_runner
export LEADERBOARD_ROOT=/cluster/work/andrebw/carla_garage/leaderboard
export PYTHONPATH="${{CARLA_ROOT}}/PythonAPI/carla/":"${{SCENARIO_RUNNER_ROOT}}":"${{LEADERBOARD_ROOT}}":${{PYTHONPATH}}
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{lib_path}
export CHALLENGE_TRACK_CODENAME=MAP''')
    rsh.write(f'''
# Server Ports
export PORT={int(gpu)*1000+start_port+iteration*10} # same as the carla server port
export TM_PORT={str(int(gpu*1000+start_port+iteration*10 + 10003))} # port for traffic manager, required when spawning multiple servers/clients

# Evaluation Setup
export ROUTES={route_path}
export SCENARIOS={scenario_path}/{scen}.json
export DEBUG_CHALLENGE=0 # visualization of waypoints and forecasting
export RESUME=1
export REPETITIONS=1
export DATAGEN=0
export BENCHMARK=collection
export GPU={str(gpu)}

# Agent Paths
export TEAM_AGENT={code_root}/team_code/{agent}.py # agent

        ''')
    if agent != 'autopilot' or agent != 'data_agent':
      rsh.write(f'''
export TEAM_CONFIG={checkpoint}

            ''')
    rsh.write(f'''
export CHECKPOINT_ENDPOINT={dataset}/Routes_{route}_Repetition{repetition}/Dataset_generation_{route}_Repetition{repetition}.json # output results file
export SAVE_PATH={dataset}/Routes_{route}_Repetition{repetition} # path for saving episodes (comment to disable)
export TM_PORT=`comm -23 <(seq {5000} {5000+400} | sort) <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`

        ''')
    rsh.write('''
export PORT=$1
echo 'World Port:' $PORT

echo "Loading modules..."
module purge
module load Anaconda3/2024.02-1
module load libjpeg-turbo/2.1.5.1-GCCcore-12.3.0

echo "Activating conda environment..."
conda activate garage
conda env list

python3 -u ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=0 \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--timeout=6000 \
--traffic-manager-port=${TM_PORT}
''')


def main():
  carla_port = 2000
  code_root = r'/cluster/work/andrebw/carla_garage_main'
  carla_root = '/cluster/work/andrebw/carla_garage/carla/CARLA_Leaderboard_20'
  data_routes_dir = '/cluster/work/andrebw/carla_garage/data/50x38_town_13'
  # Our centOS is missing some c libraries.
  # Usually miniconda has them, so we tell the linker to look there as well.
  lib_path = r'/path/to/miniconda/lib'
  date = '2024_11_06'
  dataset_name = 'hb_dataset_v01_' + date
  root_folder = r'dataset/'  # With ending slash
  data_save_root = root_folder + dataset_name
  node = 'GPUQ'
  username = 'andrebw'
  mail = 'null'
  exp_name = 'tfpp_default_expert'

  log_root = root_folder + dataset_name + '_logs'
  num_repetitions = 1
  route_paths = []
  save_folders = []
  repetitions = []
  # for i in range(num_repetitions):
  #   route_paths.append(f'{data_routes_dir}/routes/s1')
  #   route_paths.append(f'{data_routes_dir}/routes/s3')
  #   route_paths.append(f'{data_routes_dir}/routes/s4')
  #   route_paths.append(f'{data_routes_dir}/routes/s7')
  #   route_paths.append(f'{data_routes_dir}/routes/s8')
  #   route_paths.append(f'{data_routes_dir}/routes/s9')
  #   route_paths.append(f'{data_routes_dir}/routes/s10')
  #   route_paths.append(f'{data_routes_dir}/routes/ll')
  #   route_paths.append(f'{data_routes_dir}/routes/lr')
  #   route_paths.append(f'{data_routes_dir}/routes/rl')
  #   route_paths.append(f'{data_routes_dir}/routes/rr')

  #   save_folders.append(f'{data_save_root}/s1_dataset_' + date)
  #   save_folders.append(f'{data_save_root}/s3_dataset_' + date)
  #   save_folders.append(f'{data_save_root}/s4_dataset_' + date)
  #   save_folders.append(f'{data_save_root}/s7_dataset_' + date)
  #   save_folders.append(f'{data_save_root}/s8_dataset_' + date)
  #   save_folders.append(f'{data_save_root}/s9_dataset_' + date)
  #   save_folders.append(f'{data_save_root}/s10_dataset_' + date)
  #   save_folders.append(f'{data_save_root}/ll_dataset_' + date)
  #   save_folders.append(f'{data_save_root}/lr_dataset_' + date)
  #   save_folders.append(f'{data_save_root}/rl_dataset_' + date)
  #   save_folders.append(f'{data_save_root}/rr_dataset_' + date)

  #   for _ in range(11):
  #     repetitions.append(i)

  route_files = glob.glob(f"{data_routes_dir}/**/*.xml", recursive=True) 

  gpu = 0
  iteration = 0
  job_nr = 0

  # route_files = []
  # for i in range(len(route_paths)):
  #   route_path = route_paths[i]

  #   route_pattern = '*.xml'

  #   for root, _, files in os.walk(route_path):

  #     for name in files:
  #       if fnmatch.fnmatch(name, route_pattern):
  #         route_files.append(os.path.join(root, name))

  num_jobs = len(route_files)

  meta_jobs = {}
  print('Starting to generate data')
  print(f'Number of jobs: {num_jobs}')

  for i in range(len(route_files)):
    print("i", i)
    print(route_files[i])
    dataset = f'{data_save_root}/{route_files[i].split("/")[-2]}'
    # route_path = route_paths[i]
    repetition = 0#epetitions[i]

    route_pattern = '*.xml'
    checkpoint = ''
    agent = 'data_agent'

    # route_files = []
    # scen_files = []

    # for root, _, files in os.walk(route_path):
    #   for name in files:
    #     if fnmatch.fnmatch(name, route_pattern):
    #       route_files.append(os.path.join(root, name))
    #       if root.split('/')[-1].startswith('s'):
    #         scenario = root.split('/')[-1].split('s')[-1]
    #         town = name.split('Town')[-1][:2]
    #         if town == '10':
    #           town = '10HD'
    #         if os.path.exists(f'{data_routes_dir}/scenarios/s{scenario}/Town{town}_Scenario{scenario}.json'):
    #           scen_files.append(f'{data_routes_dir}/scenarios/s{scenario}/Town{town}_Scenario{scenario}.json')
    #         elif os.path.exists(os.path.join(root, f'/Town{town}_30m_Scenario{scenario}.json')):
    #           scen_files.append(os.path.join(root, f'/Town{town}_30m_Scenario{scenario}.json'))
    #         else:
    #           print(f'No scenario file for {root}')
    #           sys.exit()
    #       else:
    #         assert os.path.exists(f'{data_routes_dir}/scenarios/eval_scenarios.json'
    #                              ), f'{f"{data_routes_dir}/scenarios/eval_scenarios.json"} does not exist'
    #         scen_files.append(f'{data_routes_dir}/scenarios/eval_scenarios.json')

    bash_save_dir = Path(f'{log_root}/run_bashs')
    bash_save_dir.mkdir(parents=True, exist_ok=True)

    for ix, route in enumerate(route_files):
      route = Path(route).stem
      # scen = Path(scen_files[ix]).stem
      scenario_path = "None"#Path(scen_files[ix]).parent.absolute()
      scen = "None"

      results_save_dir = Path(f'{dataset}/Routes_{route}_Repetition{repetition}')
      results_save_dir.mkdir(parents=True, exist_ok=True)

      commands = []
      commands.append(f"export TM_PORT={str(int(gpu*1000+carla_port+iteration*10 + 10003))} # port for traffic manager, required when spawning multiple servers/clients")
      commands.append(
          f'FREE_WORLD_PORT=`comm -23 <(seq {2000} {2000 + 400} | sort) '
          f'<(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) | shuf | head -n 1`')
      commands.append("echo 'World Port:' $FREE_WORLD_PORT")
      commands.append(
          f'FREE_STREAMING_PORT=`comm -23 <(seq {3000} {3000 + 400} '
          f'| sort) <(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) | shuf | head -n 1`')
      commands.append('echo "Launching CARLA with world-port=$FREE_WORLD_PORT and streaming-port=$FREE_STREAMING_PORT"')
      carla_cmd = get_carla_command(gpu=gpu, num_try=iteration, start_port=carla_port, carla_root=carla_root)
      commands.append(f'sh {carla_root}/CarlaUE4.sh --world-port=$FREE_WORLD_PORT -RenderOffScreen -nosound -graphicsadapter=0 -carla-streaming-port=$FREE_STREAMING_PORT &')
      commands.append('sleep 10')
      create_run_eval_bash(bash_save_dir=bash_save_dir,
                           route_path=route_files[ix],
                           scenario_path=scenario_path,
                           gpu=gpu,
                           iteration=iteration,
                           route=route,
                           scen=scen,
                           repetition=repetition,
                           dataset=dataset,
                           checkpoint=checkpoint,
                           agent=agent,
                           start_port=carla_port,
                           carla_root=carla_root,
                           code_root=code_root,
                           lib_path=lib_path)
      commands.append(f'chmod u+x {bash_save_dir}/run_autopilot_{route}_Repetition{repetition}.sh')
      commands.append(f'./{bash_save_dir}/run_autopilot_{route}_Repetition{repetition}.sh $FREE_WORLD_PORT')
      commands.append('sleep 2')
      iteration += 1

      job_file = make_jobsub_file(commands=commands,
                                  job_number=job_nr,
                                  exp_name=exp_name,
                                  dataset_name=dataset_name,
                                  root_folder=root_folder,
                                  node=node,
                                  mail=mail)
      # Wait until submitting new jobs that the #jobs are at below max
      num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=exp_name, username=username, code_root=code_root)
      print(f'{num_running_jobs}/{max_num_parallel_jobs} jobs are running...')
      while num_running_jobs >= max_num_parallel_jobs:
        num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=exp_name,
                                                               username=username,
                                                               code_root=code_root)
        time.sleep(5)
      print(f'Submitting job {job_nr}/{num_jobs}: {job_file}')
      job_nr += 1

      result_file = f'{results_save_dir}/Dataset_generation_{route}_Repetition{repetition}.json'
      jobid = subprocess.check_output(f'sbatch {job_file}', shell=True).decode('utf-8').strip().rsplit(' ',
                                                                                                       maxsplit=1)[-1]
      meta_jobs[jobid] = (False, job_file, result_file, 0)
      time.sleep(0.2)  # because of automatic carla port assignment

  time.sleep(380)
  training_finished = False
  while not training_finished:
    num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=exp_name, username=username, code_root=code_root)
    print(f'{num_running_jobs} jobs are running... Job: {exp_name}')
    time.sleep(20)

    # resubmit unfinished jobs
    for k in list(meta_jobs.keys()):
      job_finished, job_file, result_file, resubmitted = meta_jobs[k]
      need_to_resubmit = False
      if not job_finished and resubmitted < 10:
        # check whether job is running
        if int(subprocess.check_output(f'squeue | grep {k} | wc -l', shell=True).decode('utf-8').strip()) == 0:
          # check whether result file is finished?
          if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f_result:
              evaluation_data = json.load(f_result)
            progress = evaluation_data['_checkpoint']['progress']
            if len(progress) < 2 or progress[0] < progress[1]:
              need_to_resubmit = True
            else:
              for record in evaluation_data['_checkpoint']['records']:
                if record['scores']['score_route'] <= 0.00000000001:
                  need_to_resubmit = True
                if record['status'] == 'Failed - Agent couldn\'t be set up':
                  need_to_resubmit = True
                if record['status'] == 'Failed':
                  need_to_resubmit = True
                if record['status'] == 'Failed - Simulation crashed':
                  need_to_resubmit = True
                if record['status'] == 'Failed - Agent crashed':
                  need_to_resubmit = True

            if not need_to_resubmit:
              # delete old job
              print(f'Finished job {job_file}')
              meta_jobs[k] = (True, None, None, 0)

          else:
            need_to_resubmit = True

      if need_to_resubmit:
        print(f'resubmit sbatch {job_file}')

        # rename old error files to still access it
        job_nr_log = Path(job_file).stem
        time_now_log = time.time()
        os.system(f'mkdir -p "{log_root}/run_files/logs_{time_now_log}"')
        os.system(f'cp {log_root}/run_files/logs/qsub_err{job_nr_log}.log {log_root}/run_files/logs_{time_now_log}')
        os.system(f'cp {log_root}/run_files/logs/qsub_out{job_nr_log}.log {log_root}/run_files/logs_{time_now_log}')

        jobid = subprocess.check_output(f'sbatch {job_file}', shell=True).decode('utf-8').strip().rsplit(' ',
                                                                                                         maxsplit=1)[-1]
        meta_jobs[jobid] = (False, job_file, result_file, resubmitted + 1)
        meta_jobs[k] = (True, None, None, 0)

    time.sleep(10)

    if num_running_jobs == 0:
      training_finished = True


if __name__ == '__main__':
  main()
