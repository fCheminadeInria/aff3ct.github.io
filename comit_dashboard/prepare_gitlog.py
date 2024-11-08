import pandas as pd
import panel as pn
from datetime import datetime
import os
import csv
import re

#prépare la base de données Git
# la dernière colonne du log de Git contient des ",".
def format_git_log(git_version_csv_path, key) :

    with open(git_version_csv_path, "r", newline="") as fp:
        reader = csv.reader(fp, delimiter=",")
        rows = [x[:4] + [','.join(x[4:-1])] for x in reader] 
        git_df = pd.DataFrame(rows)
        
    new_header = git_df.iloc[0] #grab the first row for the header
    git_df = git_df[1:] #take the data less the header row
    git_df.columns = new_header #set the header row as the df header
    
    git_df["Project"] = key
    
    return git_df

data_input_path  = "./comit_dashboard/database/data_from_scratch/"
database_path    = "./comit_dashboard/database/"

log_affect_git_path    = data_input_path + 'log_commit_aff3ct.csv'  
log_streampu_git_path  = data_input_path + 'log_commit_streampu.csv'  

df_git_aff3ct = format_git_log(log_affect_git_path  , "Aff3ct")
df_git_spu    = format_git_log(log_streampu_git_path, "StreamPu")




# Fusion des tables
df_git = pd.concat([df_git_aff3ct, df_git_spu], ignore_index=True)

# Conversion au format datetime
df_git['date'] = pd.to_datetime(df_git['date'],utc=True, errors='coerce')
df_git['date'] = df_git['date'].dt.tz_localize(None)

# Ordonancement des colonnes
df_git = df_git.reindex(['Project','date','contributor', 'echo sha', 'message' ], axis=1)

# supression des virgule pour l'export csv
df_git['message'] = df_git['message'].str.replace(',', ';')
df_git.to_csv(database_path + 'log_git.csv', index=False)

# Chargment des autres fichiers
config_csv_path = data_input_path + 'config.csv'
task_csv_path = data_input_path + 'task_noise.csv'
performance_csv_path = data_input_path + 'performance_noise.csv'
config_df       = pd.read_csv (config_csv_path)
task_df         = pd.read_csv (task_csv_path)
performance_df  = pd.read_csv (performance_csv_path)

# Préparation
config_df = config_df.rename(columns={'Meta.Command Line': 'Meta.Command', 'Meta.Git version' : 'Meta.GitVersion'})


# création de la commande courte

def extract_command_short(command):
    # Étape 1 : Extraire le chemin vers l'exécutable et le nom
    command = re.sub(r'^[^\s]+', '', command)

    # Étape 2 : Retirer les options spécifiées
    # Regex pour retirer les options avec ou sans arguments
    options_to_remove = [
        r'--sim-stats',
        r'--mnt-mutinfo',
        r'--ter-no',
        r'--sim-json\s+"[^"]*"'  # Option suivie d'un chemin entre guillemets
    ]
    for option in options_to_remove:
        command = re.sub(option, '', command)

    # Étape 3 : Nettoyer les espaces superflus
    command = re.sub(r'\s+', ' ', command).strip()

    return command

# Appliquer la fonction à la colonne Meta.Command
config_df['Meta.Command_short'] = config_df['Meta.Command'].apply(extract_command_short)



task_df = task_df[task_df.Module != "TOTAL"]
task_df = task_df.merge(performance_df[['Config_Hash', 'Noise_Level', 'Signal Noise Ratio(SNR).Eb/N0(dB)','Signal Noise Ratio(SNR).Es/N0(dB)','Signal Noise Ratio(SNR).Sigma']], on=['Config_Hash', 'Noise_Level'],
    how='left')

# Export au format csv
config_df       .to_csv(database_path + 'config.csv'      , index=False)
task_df         .to_csv(database_path + 'tasks.csv'       , index=False)
performance_df  .to_csv(database_path + 'performances.csv', index=False)
