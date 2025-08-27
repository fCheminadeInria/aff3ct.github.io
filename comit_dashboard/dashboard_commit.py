from bokeh.settings import settings
settings.resources = 'inline'

import pandas as pd
import panel as pn
import plotly.graph_objs as go
import plotly.express as px
import plotly.graph_objects as go
import param
import unicodedata as ud
from io import BytesIO
import sys
import os
import unicodedata as ud
import urllib.request

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

# ------------------------------------------------------------------------------
#  Variables d’environnement
# ------------------------------------------------------------------------------
IS_PYODIDE       = sys.platform == "emscripten"
IS_PANEL_CONVERT = os.getenv("PANEL_CONVERT") == "1"
GITLAB_PACKAGE_URL = "https://gitlab.inria.fr/api/v4/projects/1420/packages/generic/gitlab-elk-export/latest/export/"
BUTON_WIDTH = 60
# ------------------------------------------------------------------------------
#  Chargement des données – SYNCHRONE
# ------------------------------------------------------------------------------

def load_table(name: str, fmt: str = "parquet") -> pd.DataFrame:
    
    url = f"{GITLAB_PACKAGE_URL}{name}.{fmt}"
    CHUNK = 1024 * 1024          # 1 Mo par appel

    headers = {
        "User-Agent": "Mozilla/5.0",
    }

    all_data = BytesIO()
    start = 0

    while True:
        headers["Range"] = f"bytes={start}-{start + CHUNK - 1}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=None) as resp:
                data = resp.read()
                if not data:          # plus rien
                    break
                all_data.write(data)
                if len(data) < CHUNK: # dernier morceau
                    break
                start += len(data)
        except urllib.error.HTTPError as e:
            if e.code == 416:         # Range Not Satisfiable → fin atteinte
                break
            else:
                raise

    all_data.seek(0)
    if fmt == "json":
        df = pd.read_json(all_data, lines=True)
        if len(df) == 1 and df.isna().all(axis=None):
            # Retrait de la ligne factice
            empty_df = df.iloc[:0].reset_index(drop=True)
            empty_df = empty_df.astype("string")  # Force un type neutre
            return empty_df
        return df
    elif fmt == "parquet":
        return pd.read_parquet(all_data)
    else:
        return pd.DataFrame()

def load_data_sync() -> None:
    """Charge toutes les tables dans pn.state.cache['db'] (synchrone)."""
    print("⚙️ load_data_sync() appelé")
    fmt='json'
    
    tables = [
        'git',
        'command',
        'parameters',
        'exec',
        'runs',
    ]
    db = dict()
    for table in tables:
        db[table] = load_table(table, fmt)
        print(f"{tables.index(table) + 1}/{len(tables)} {table} chargé : {len(db[table])} lignes")

    if db['command'].empty:
        raise ValueError("Impossible de charger les données pour 'command'. Veuillez vérifier l'URL et les dépendances.")
    db['command'].set_index('Command_id', inplace=True)

    db['parameters'].set_index('param_id', inplace=True)
    
    db['runs'].set_index('RUN_id', inplace=True)
    
    db['git'].set_index('sha1', inplace=True)

    # Alias
    db['command']['Config_Alias'] = (
        db['command'].index.astype(str) + " : " +
        db['command']['Command_short'].astype(str)
    )
    
    db['exec'].set_index('log_hash', inplace=True)
    db['exec']['Date_Execution'] = pd.to_datetime(db['exec']['Date_Execution'], unit='ms', errors='coerce')
    db['exec']["alias"] = db['exec']['EXEC_id'] + " : " + db['exec']['Date_Execution'].dt.strftime('%Y-%m-%d %H:%M:%S, (Sha1 : ') +  db['exec']['sha1']  + ")"

    pn.state.cache['db'] = db

    apply_typing_code()
        
    print("✅ load_data_sync() terminé")
    
# ------------------------------------------------------------------------------
#  Typage automatique (copié-collé de la version JSON)
# ------------------------------------------------------------------------------

def apply_typing_code():
    ''' Applique le typage des données  (copier coller du résultat de generate_typing_code) ''' 
    # Typage pour runs
    runs = pn.state.cache['db']['runs']
    
    to_convert = {
        'Bit Error Rate (BER) and Frame Error Rate (FER).BE': 'Int64',
        'Bit Error Rate (BER) and Frame Error Rate (FER).BER': 'float',
        'Bit Error Rate (BER) and Frame Error Rate (FER).FE': 'Int64',
        'Bit Error Rate (BER) and Frame Error Rate (FER).FER': 'float',
        'Bit Error Rate (BER) and Frame Error Rate (FER).FRA': 'Int64',
        'Global throughputand elapsed time.SIM_THR(Mb/s)': 'float',
        'Global throughputand elapsed time.elapse_time(ms)': 'float',
        'Signal Noise Ratio(SNR).Eb/N0(dB)': 'float',
        'Signal Noise Ratio(SNR).Es/N0(dB)': 'float',
        'Signal Noise Ratio(SNR).Sigma': 'float',
        'Signal Noise Ratio(SNR).Event Probability': 'float',
        'Signal Noise Ratio(SNR).Received Optical': 'float',
    }
    
    for col, dtype in to_convert.items():
        runs[col] = pd.to_numeric(runs[col], errors='coerce')
        if dtype == 'Int64':
            runs[col] = runs[col].astype('Int64')
            
            
    pn.state.cache['db']['runs'] = runs

    # Typage pour git
    git = pn.state.cache['db']['git']
    git['author'] = git['author'].astype(str)
    git['email'] = git['email'].astype(str)
    git['date'] = pd.to_datetime(git['date'], errors='coerce')
    git['message'] = git['message'].astype(str)
    git['insertions'] = pd.to_numeric(git['insertions'], errors='coerce').astype('Int64')
    git['deletions'] = pd.to_numeric(git['deletions'], errors='coerce').astype('Int64')
    git['files_changed'] = pd.to_numeric(git['files_changed'], errors='coerce').astype('Int64')
    pn.state.cache['db']['git'] = git

# ------------------------------------------------------------------------------
#  Initialisation du dashboard
# ------------------------------------------------------------------------------
def init_dashboard():
    print("⚙️ init_dashboard() appelé")

    db = pn.state.cache['db']

    git_model = GitFilterModel(df_git=db['git'])

    command_model = CommandFilterModel(git_filter=git_model)
    panelCommit = PanelCommit(command_filter=command_model, git_filter=git_model)

    lvl2_model = Lvl2_Filter_Model(command_filter=command_model)
    config_panel = Lvl2_ConfigPanel(lv2_model=lvl2_model)

    panelConfig = pn.Row(
        pn.Column(
            TableConfig(lvl2_model=lvl2_model),
            config_panel,
            Lvl2_Git_Selector(lv2_model=lvl2_model),
            pn.Tabs(
                ('BER/FER', PerformanceBERFERPlot(lvl2_model = lvl2_model, noise_scale_param=noiseScale))),
            sizing_mode="stretch_width"
        )
    )

    lvl3 = Level3(lvl2_model=lvl2_model)

    panelData = pn.Column(
        pn.indicators.Number(
        name="Commits en base",
        value=db['git'].shape[0] if not db['git'].empty else 0
        ),
        pn.indicators.Number(
        name="Commandes en base",
        value=db['command'].shape[0] if not db['command'].empty else 0
        ), 
        pn.indicators.Number(
            name="Executions en base",
            value=db['exec'].shape[0] if not db['exec'].empty else 0
        ),
        RunsCountIndicator(git_filter=git_model), 
        sizing_mode="stretch_width")

    dashboard = pn.Column(
        pn.pane.HTML("<h2>✏️ Niveau 1 : Evolution par commit</h2>"),
        panelCommit,
        pn.pane.HTML("<h2>☎️ Niveau 2 : BER / FER</h2>"),
        panelConfig,
        pn.pane.HTML("<h2>⚙️ Niveau 3 : Analyse par exécutions</h2>"),
        lvl3,
        sizing_mode="stretch_width"
    )

    logo = pn.pane.Image(
        "https://raw.githubusercontent.com/fCheminadeInria/aff3ct.github.io/"
        "refs/heads/master/comit_dashboard/image/93988066-1f77-4b42-941f-1d5ef89ddca2.webp",
        width=200
    )

    template = pn.template.FastListTemplate(
        title="Tableau de bord de suivi des commits",
        sidebar=[logo, noiseScale, pn.layout.Divider(), panelData],
        main=[dashboard],
        main_layout=None,
        accent="teal",
        theme_toggle=False,
    )

    print("✅ init_dashboard() terminé")
    return template

# ------------------------------------------------------------------------------
#  Fonctions utilitaires
# ------------------------------------------------------------------------------

def command_alias_to_command_id(alias: str) -> str:
    """
    Convertit un alias de configuration en Command_id.
    
    Args:
        alias (str): L'alias de la configuration.
        
    Returns:
        str: Le Command_id correspondant ou None si non trouvé.
    """
    db = pn.state.cache.get('db', {})
    df = db.get('command', pd.DataFrame())
    
    if df.empty or 'Config_Alias' not in df.columns:
        return None
    
    match = df[df['Config_Alias'] == alias]
    if not match.empty:
        return match.index[0]  # Renvoie le Command_id
    else:
        return None

def command_id_to_alias(command_id: str) -> str:
    """
    Convertit un Command_id en alias de configuration.
    
    Args:
        command_id (str): Le Command_id de la configuration.
        
    Returns:
        str: L'alias de la configuration ou None si non trouvé.
    """
    db = pn.state.cache.get('db', {})
    df = db.get('command', pd.DataFrame())
    
    if df.empty or 'Config_Alias' not in df.columns:
        return None
    
    if command_id in df.index:
        return df.loc[command_id, 'Config_Alias']
    else:
        return None

def exec_alias_to_log_hash(alias: str) -> str:
    """
    Convertit un alias d'exécution en log_hash.
    
    Args:
        alias (str): L'alias de l'exécution.
        
    Returns:
        str: Le log_hash correspondant ou None si non trouvé.
    """
    db = pn.state.cache.get('db', {})
    df_exec = db.get('exec', pd.DataFrame())
    
    if df_exec.empty or 'alias' not in df_exec.columns:
        return None
    
    match = df_exec[df_exec['alias'] == alias]
    if not match.empty:
        return match.index[0]  # Renvoie le log_hash
    else:
        return None
    
def exec_log_hash_to_alias(log_hash: str) -> str:
    """
    Convertit un log_hash en alias d'exécution.
    
    Args:
        log_hash (str): Le log_hash de l'exécution.
        
    Returns:
        str: L'alias de l'exécution ou None si non trouvé.
    """
    db = pn.state.cache.get('db', {})
    df_exec = db.get('exec', pd.DataFrame())
    
    if df_exec.empty or 'alias' not in df_exec.columns:
        return None
    
    if log_hash in df_exec.index:
        return df_exec.loc[log_hash, 'alias']
    else:
        return None    

##################################### Niveau Global ####################################

IS_PYODIDE       = sys.platform == "emscripten"
IS_PANEL_CONVERT = os.getenv("PANEL_CONVERT") == "1"

# Initialiser Panel
pn.extension(
    "plotly", 
    sizing_mode="stretch_width", 
    )

##################################### Variable statiques ####################################
noise_label = {
    'Eb/N0': 'Signal Noise Ratio(SNR).Eb/N0(dB)',
    'Es/N0': 'Signal Noise Ratio(SNR).Es/N0(dB)',
    'Sigma': 'Signal Noise Ratio(SNR).Sigma',
}

######################
## Echelle de bruit ##
######################

class NoiseScale (pn.viewable.Viewer) :
    value = param.String(default= 'Signal Noise Ratio(SNR).Eb/N0(dB)', allow_refs=True)
    noise_label = param.Dict()
    
    def __init__ (self, **params):
        super().__init__(**params)
        
        self.radio_group = pn.widgets.RadioBoxGroup(
            name='Echelle de bruit', 
            options=list(self.noise_label.keys()), 
            value=list(self.noise_label.keys())[0], 
            inline=True )
        self._update_value(None)
        self.radio_group.param.watch(self._update_value, "value")

    def __panel__(self):
        return pn.Column(
            pn.pane.Markdown(f"**{self.radio_group.name} :** "),
            pn.Row(self.radio_group, css_classes=["align-right"],sizing_mode="stretch_width"),
            sizing_mode="stretch_width")

    def _update_value(self, event):
        """
        Met à jour la propriété `value` en fonction de la sélection.
        """
        self.value = self.noise_label[self.radio_group.value]

##################################### Modèle de données ####################################

##################################
## Gestion des données niveau 1 ##
##################################

class GitFilterModel(param.Parameterized):
    """Modèle pour filtrer les données Git par plage de dates."""
    df_git = param.DataFrame()
    date_range = param.Tuple(default=(None, None), length=2, doc="Plage de dates pour filtrer")

    def __init__(self, **params):
        super().__init__(**params)
        # Si date_range n'est pas fourni, on initialise avec la plage complète des dates
        db = pn.state.cache['db']
        if db['runs'].empty:
            loaded_runs=[]
        else:
            loaded_runs = db['runs']['log_hash'].tolist()
        loaded_sha1 = db['exec'][db['exec'].index.isin(loaded_runs)]['sha1'].tolist()
        
        git_exec = db['git'][db['git'].index.isin(loaded_sha1)]
        
        if self.date_range == (None, None):
            min_date = git_exec['date'].min() if not git_exec.empty else db ['git']['date'].max()
            max_date = db ['git']['date'].max()
            self.date_range = (min_date, max_date)

    @param.depends('date_range', watch=True)
    def set_filtered_df(self):
        db = pn.state.cache['db']
        df = db ['git'].copy()
        start, end = self.date_range
        if start and end:
            start = pd.to_datetime(start).floor('s')
            end   = pd.to_datetime(end).ceil('s')
            df = df[(df['date'] >= start) & (df['date'] <= end)]
        self.df_git = df
    
    def get_sha1_valids(self):
        return self.df_git.index.unique()

    @param.depends('df_git', watch=True)
    def load_runs_from_commit(self):
        """Charge les runs associées à une liste de commits s'ils ne sont pas déjà en mémoire."""

        db = pn.state.cache['db']

        # 1. SHA1 déjà chargés (on regarde les runs déjà présents et on récupère les sha1 correspondants)
        if db['runs'].empty:
            loaded_runs =[]
        else:    
            loaded_runs = db['runs']['log_hash'].tolist()
        loaded_sha1 = db['exec'][db['exec'].index.isin(loaded_runs)]['sha1'].tolist()

        # 2. SHA1 à charger = différence parmi ceux qui ont une execution
        sha1_to_load = self.get_sha1_valids().difference(loaded_sha1)
        sha1_to_load = sha1_to_load.intersection(db['exec']['sha1'])

        for sha1 in sha1_to_load:
            df_runs_part = load_table(f"runs/runs_{sha1}", fmt='json')
            if df_runs_part.empty:
                continue  # on passe au suivant

            df_runs_part.set_index('RUN_id', inplace=True)

            # 3. Conversion des colonnes numériques
            to_convert = {
                'Bit Error Rate (BER) and Frame Error Rate (FER).BE': 'Int64',
                'Bit Error Rate (BER) and Frame Error Rate (FER).BER': 'float',
                'Bit Error Rate (BER) and Frame Error Rate (FER).FE': 'Int64',
                'Bit Error Rate (BER) and Frame Error Rate (FER).FER': 'float',
                'Bit Error Rate (BER) and Frame Error Rate (FER).FRA': 'Int64',
                'Global throughputand elapsed time.SIM_THR(Mb/s)': 'float',
                'Global throughputand elapsed time.elapse_time(ms)': 'float',
                'Signal Noise Ratio(SNR).Eb/N0(dB)': 'float',
                'Signal Noise Ratio(SNR).Es/N0(dB)': 'float',
                'Signal Noise Ratio(SNR).Sigma': 'float',
                'Signal Noise Ratio(SNR).Event Probability': 'float',
                'Signal Noise Ratio(SNR).Received Optical': 'float',
            }

            for col, dtype in to_convert.items():
                df_runs_part[col] = pd.to_numeric(df_runs_part[col], errors='coerce')
                if dtype == 'Int64':
                    df_runs_part[col] = df_runs_part[col].astype('Int64')

            # 5. Ajout au DataFrame principal
            db['runs'] = pd.concat([db['runs'], df_runs_part])

        # 6. Mise à jour du cache
        pn.state.cache['db'] = db

class CommandFilterModel(param.Parameterized):
    """Modèle pour filtrer les commandes sur la base du filtrage de Git selon le code."""
    git_filter = param.ClassSelector(class_=GitFilterModel) 
    code = param.ListSelector(default=[], objects=[])
    df_exec = param.DataFrame(default=pd.DataFrame(), doc="DataFrame des exécutions filtrées")

    def __init__(self, **params):
        super().__init__(**params)
        
        db = pn.state.cache['db']
        
        # Initialisation de 'code' avec toutes les valeurs possibles dans df_commands['code']
        all_codes = sorted(db['command']['code'].dropna().unique().tolist())
        self.param['code'].objects = all_codes
        self.param['code'].default = all_codes 
        self.code = all_codes
        self._df_exec()
    
    @param.depends('git_filter.df_git', 'code', watch=True)
    def _df_exec(self):
        """
        Filtre les executions selon le modèle (code et sha1).
        
        returns:   
            DataFrame filtrée des exécutions avec les commandes correspondantes. (avec le code issue de db['command'])
        """
        db = pn.state.cache['db'] 
        
        # filtre les exécutions par les sha1 valides du git_filter
        df_exec = db['exec'][db['exec']['sha1'].isin(self.sha1)]     
        
        # filtre les commandes par les codes sélectionnés
        commands = self.df_commands
        
        self.param.code.objects  = sorted(commands['code'].dropna().unique().tolist())
        
        
        index_name = df_exec.index.name
        # ajoute les command_id à exec élimine les lignes sans correspondance de command_id
        df_exec = (
            df_exec
            .reset_index()  
            .merge(
                commands[['code']],
                left_on='Command_id',
                right_index=True,
                how='inner'
            )
            .set_index('log_hash')
        )
                
        self.df_exec =  df_exec
        self.param.trigger('df_exec')
        
    @property
    def df_commands(self):
        db = pn.state.cache['db'] 
        return db['command'][db['command']['code'].isin(self.code)]
    
    @property
    def sha1(self):
        """
        Renvoie les sha1 sélectionnés.
        """
        return self.git_filter.get_sha1_valids()
    
    @property
    def commands(self):
        """Renvoie les commandes filtrées par le modèle.

        Returns:
            list: identifiants des commandes filtrées.
        """
        return self.df_commands.index.tolist()
    
    
################################################
## Gestion des données niveau 2 avec filtrage ##
################################################
class Lvl2_Filter_Model(param.Parameterized):
    command_filter = param.ClassSelector(class_=CommandFilterModel)
    value_commands = param.List(default=[])          # liste d’index (command_id) sélectionnés via les commits
    options = param.DataFrame()             # DataFrame filtré par le niveau 1 : index = command_id, colonnes = ['sha1', 'Config_Alias', ...]
    value_sha1 = param.List(default=[])          # liste d’index (sha1) sélectionnés via les commits
    
    def __init__(self, **params):
        super().__init__(**params)
        # On observe le DataFrame filtré du niveau 1
        self.value_commands = []
        self.value_sha1 = []
        self._update_from_lvl1()

    @param.depends('command_filter.df_exec', watch=True)
    def _update_from_lvl1(self, *events):
        """Met à jour les options et la sélection en fonction du filtrage du niveau 1."""
        self.options = self.command_filter.df_exec.copy().merge(
            self.command_filter.df_commands[['Config_Alias']],
            left_on='Command_id',
            right_index=True,
            how='inner'
        )

        # Nettoie la sélection courante
        self.value_commands = list(dict.fromkeys(
            v for v in self.value_commands if v in self.command_filter.commands
        ))
        
        self.value_sha1 = list(dict.fromkeys(
            v for v in self.value_sha1 if v in self.command_filter.sha1
        ))

    @property
    def df_commands(self):
        """DataFrame filtré par le niveau 2 : index = command_id, colonnes = ['sha1', 'Config_Alias', ...]"""
        db = pn.state.cache['db']
        return db['command'][db['command'].index.isin(self.cross_commands)].copy()

    @property
    def cross_commands(self):
        """Renvoie les commandes sélectionnées."""
        return self.df_exec['Command_id'].unique().tolist()

    @property
    def cross_sha1(self):
        """Renvoie les sha1 sélectionnés."""
        return self.df_exec['sha1'].unique().tolist()

    @property
    def df_exec(self):
        """DataFrame des exécutions (runs) filtrées par le niveau 2."""
        exec_df = self.command_filter.df_exec
        exec_df = exec_df[exec_df['Command_id'].isin(self.value_commands)]
        exec_df = exec_df[exec_df['sha1'].isin(self.value_sha1)]
        return exec_df

    @property
    def df_runs(self):
        runs = pn.state.cache['db']['runs']  
        
        sel = runs[runs["log_hash"].isin(self.df_exec.index)]
        # ajoute la colonne sha1 issue du DataFrame self.df
        sel = (
            sel.reset_index()
            .merge(
                self.df_exec[['sha1', 'Command_id']],
                left_on='log_hash',
                right_index=True,
                how='inner'
            )
            .set_index(runs.index.name)
        )
            
        return sel

    def reset_commands(self):
        self.value_commands = []
        self._update_from_lvl1()
        
    def reset_sha1(self):
        self.value_sha1 = []
        self._update_from_lvl1()

##################################################
## Gestion des données niveau 3 : config unique ##
##################################################

class ConfigUniqueModel(param.Parameterized):
    lv2_model = param.ClassSelector(default=None, class_=Lvl2_Filter_Model)
    config = param.Selector(default=None, objects=[])
    options = param.ListSelector(default=[], objects=[])

    @property
    def df(self):
        if self.config is None:
            return self.lv2_model.df_exec.iloc[0:0]  # DataFrame vide
        return self.lv2_model.df_exec.loc[self.config]

    @property
    def df_runs(self):
        db = pn.state.cache.get('db', {})
        if self.config is None:
            return pd.DataFrame()
        return  db['runs'][db['runs']['log_hash'].isin(self.df_exec.index)]        
 
    @property
    def options_alias(self):
        return self.lv2_model.df_commands['Config_Alias'].tolist()

    def alias(self):
        if self.config is None or self.config not in self.lv2_model.df_exec['Command_id'].values :
            return None
        return command_id_to_alias (self.config)

    @param.depends('lv2_model.value_commands', 'lv2_model.value_sha1', watch=True)
    def _on_lvl2_df_change(self):
        # Initialise la valeur avec le command_id correspondant au premier alias
        self.options = self.lv2_model.df_commands.index.tolist()

    @property
    def df_exec(self):
        """Renvoie le DataFrame des exécutions filtrées par la configuration unique."""
        df_exec = self.lv2_model.df_exec
        return df_exec[df_exec['Command_id'] == self.config].copy()
    
class ExecUniqueModel(param.Parameterized):
    """
    Modèle pour gérer un exécution unique (lot de run de SNR différents) d'une configuration spécifique.
    """
    unique_conf_model = param.ClassSelector(class_=ConfigUniqueModel)
    log_hash = param.Selector(default=None, objects=[])   # valeur réelle (hash)
    trig_opt_changed = param.Event()

    # Mise à jour automatique quand le modèle parent change
    @param.depends('unique_conf_model.config', 'unique_conf_model.lv2_model.value_sha1', watch=True)
    def _update_exec(self):
        """Construit la liste des exécutions disponibles et met à jour le sélecteur."""
        df_exec = self.unique_conf_model.df_exec
        
        if df_exec.empty:
            self.param['log_hash'].objects = {None: None}
            self.log_hash = None
        else:
            #mise à jour des options
            self.param['log_hash'].objects = df_exec.index.to_list()
            # Sélectionner la première exécution par défaut
            if self.log_hash is None or self.log_hash not in self.param['log_hash'].objects[0]:
                self.log_hash = self.param['log_hash'].objects[0]    
        self.trig_opt_changed = True
            
    @property
    def df_runs(self):
        """Renvoie le sous-ensemble des runs (SNR différents) pour la config sélectionnée."""
        if self.unique_conf_model is None or self.log_hash is None:
            return pd.DataFrame()
        df = self.unique_conf_model.df_runs
        return df[df['log_hash'] == self.log_hash]

    @property
    def df_tasks(self):
        """Charge les tâches associées à un exec unique."""
        if self.log_hash is None:
            return pd.DataFrame()

        table_name = f"tasks/{self.log_hash}"
        try:
            df_tasks = load_table(table_name, fmt='json')
            if df_tasks.empty:
                return pd.DataFrame()
        except Exception as e:
            print(f"⚠️ Impossible de charger {table_name}: {e}")
            return pd.DataFrame()

        # Ajoute les colonnes de bruit depuis le df_runs
        df_runs = self.df_runs
        noise_cols = list(noise_label.values())

        df_noise = df_runs[df_runs['log_hash'] == self.log_hash][noise_cols]
        df_tasks = (df_tasks
                    .set_index('RUN_id')
                    .join(df_noise, how='inner')
                    .reset_index())
        return df_tasks

    @property
    def options(self):
        """Liste des labels d'exécution disponibles pour la sélection."""
        return self.unique_conf_model.df_exec['alias'].tolist()

    @property
    def log(self):
        """Retourne le contenu du log associé au run_id (pas basé sur date)."""
        if self.log_hash is None:
            return "```Aucun run sélectionné.```"
        return f"```\n{self.__load_log()}\n```"

    def __load_log(self) -> str:
        """Lit un fichier distant hébergé sur GitLab."""
        CHUNK = 1024 * 1024
        url = f"{GITLAB_PACKAGE_URL}logs/{self.log_hash}.log"

        headers = {"User-Agent": "Mozilla/5.0"}
        all_data = BytesIO()
        start = 0

        while True:
            headers["Range"] = f"bytes={start}-{start + CHUNK - 1}"
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=None) as resp:
                    data = resp.read()
                    if not data:
                        break
                    all_data.write(data)
                    if len(data) < CHUNK:
                        break
                    start += len(data)
            except urllib.error.HTTPError as e:
                if e.code == 416:
                    break
                elif e.code == 404:
                    return "❌ Erreur : Fichier introuvable."
                else:
                    return f"❌ Erreur HTTP : {e.code} - {e.reason}"
            except Exception as e:
                return f"❌ Erreur inattendue : {str(e)}"

        return all_data.getvalue().decode('utf-8', errors='replace')

##################################### Niveau 1 : Git et perf global ####################################

#################################
## Component pour le Panel Git ##
#################################
class DateRangeFilter(pn.viewable.Viewer):
    git_filter = param.ClassSelector(class_=GitFilterModel)

    def __init__(self, **params):
        super().__init__(**params)
        # Bornes extraites du DataFrame Git
        df = pn.state.cache['db']['git']
        
        start, end = df['date'].min(), df['date'].max()
        
        # Création du slider
        self.slider = pn.widgets.DatetimeRangeSlider(
            name='Filtre sur les dates des commits',
            start=start,
            end=end,
            value=self.git_filter.date_range,
            sizing_mode='stretch_width',
            step = 300,
        )
        
        self.slider.param.watch(
            lambda event: setattr(self.git_filter, 'date_range', event.new),
            'value_throttled'
        )


    def __panel__(self):
        return self.slider

class PerformanceByCommit(pn.viewable.Viewer):
    command_filter = param.ClassSelector(class_=CommandFilterModel)

    def __init__(self, **params):
        super().__init__(**params)

        self.plot_throughput_pane = pn.pane.Plotly(sizing_mode='stretch_width')
        self.plot_latency_pane = pn.pane.Plotly(sizing_mode='stretch_width')

        self.tabs = pn.Tabs(
            ('⏱️ Latence', self.plot_latency_pane),
            ('⚡ Débit', self.plot_throughput_pane),
            dynamic=True
        )

        self.command_filter.param.watch(self._update_all, ['df_exec'])

        self._update_all()

    def __panel__(self):
        return self.tabs

    def _update_all(self, *events):
        self._update_data()
        self._create_plots()

    def _update_data(self):
        throughput_col = 'Global throughputand elapsed time.SIM_THR(Mb/s)'
        latency_col = 'Global throughputand elapsed time.elapse_time(ms)'

        db = pn.state.cache['db']
        df_exec = self.command_filter.df_exec

        if df_exec.empty:
            # Crée un DataFrame vide avec colonnes attendues
            self.df_grouped = pd.DataFrame(columns=['sha1', 'Code', 'Débit moyen (Mb/s)', 'Latence moyenne (ms)', 'date'])
            return

        df_exec = df_exec.merge(
            db['git'][['date']], left_on='sha1', right_index=True, how='inner'
        )
        df = db['runs'].merge(
            df_exec[['Command_id', 'sha1', 'date', 'code']], on='log_hash', how='left'
        ).sort_values(by='date')

        self.df_grouped = df.groupby(['sha1', 'code']).agg({
            throughput_col: 'mean',
            latency_col: 'mean',
            'date': 'first'
        }).reset_index().rename(columns={
            throughput_col: 'Débit moyen (Mb/s)',
            latency_col: 'Latence moyenne (ms)',
            'code': 'Code',
        }).sort_values(by='date')

    def _create_plots(self):
        self.plot_throughput_pane.object = px.line(
            self.df_grouped,
            x='date', y='Débit moyen (Mb/s)',
            color='Code',
            title="Débit moyen par commit (par code)",
            markers=True
        ).update_layout(
            legend=dict(orientation='v', y=1, x=1.05),
            margin=dict(r=100),
            xaxis=dict(title="Date", rangeslider=dict(visible=True), showgrid=True),
            yaxis=dict(title="Débit moyen (Mb/s)", showgrid=True),
        )
        if self.df_grouped.empty:
            self.plot_throughput_pane.object.add_annotation(text="Aucune donnée sélectionnée",
                                                    xref="paper", yref="paper",
                                                    x=0.5, y=0.5, showarrow=False)


        self.plot_latency_pane.object = px.line(
            self.df_grouped,
            x='date', y='Latence moyenne (ms)',
            color='Code',
            title="Latence moyenne par commit (par code)",
            markers=True
        ).update_layout(
            legend=dict(orientation='v', y=1, x=1.05),
            margin=dict(r=100),
            xaxis=dict(title="Date", rangeslider=dict(visible=True), showgrid=True),
            yaxis=dict(title="Latence moyenne (ms)", showgrid=True),
        )
        if self.df_grouped.empty:
            self.plot_latency_pane.object.add_annotation(text="Aucune donnée sélectionnée",
                                                    xref="paper", yref="paper",
                                                    x=0.5, y=0.5, showarrow=False)


##########################
## Sélecteur de code ##
##########################
class CodeSelector(pn.viewable.Viewer):
    cmd_filter_model = param.ClassSelector(class_=CommandFilterModel)

    def __init__(self, **params):
        super().__init__(**params)
        self.widget = pn.widgets.CheckBoxGroup(name='Codes à afficher', inline=True)
        
        db = pn.state.cache['db']
        self.widget.options = self.cmd_filter_model.param.code.objects
        self.widget.value = self.cmd_filter_model.param['code'].default  # Affecte la valeur par défaut des codes     
        self.widget.param.watch(self._update_filter, 'value')
        
        self.select_all_button = pn.widgets.Button(name='Reset', button_type='primary', width = BUTON_WIDTH)
        self.select_all_button.on_click(self.select_all_codes)

    def select_all_codes(self, event):
        self.widget.value = self.widget.options

    def _update_filter(self, event):
        self.cmd_filter_model.code = self.widget.value
        
    def __panel__(self):
        return pn.Row(self.widget, self.select_all_button, sizing_mode="stretch_width")
    
##############################
## Table des commits Git ##
##############################

class FilteredTable(pn.viewable.Viewer):
    filter_model = param.ClassSelector(class_=GitFilterModel)

    def __init__(self, **params):
        super().__init__(**params)
        self.table = pn.widgets.DataFrame(height=300, text_align='center', sizing_mode="stretch_width")
        self._update()

    @param.depends('filter_model.df_git', watch=True)
    def _update(self, *events):
        self.table.value = self.filter_model.df_git

    def __panel__(self):
        return self.table

#####################
## Indicateurs Git ##
#####################

class GitIndicators(pn.viewable.Viewer):
    filter_model = param.ClassSelector(class_=GitFilterModel)

    def __init__(self, **params):
        super().__init__(**params)

        self.commit_count = pn.indicators.Number(name="Commits sélectionnés", value=0)
        self.git_version_count = pn.indicators.Number(name="avec données", value=0)
        self.last_commit_text = pn.widgets.StaticText(name="Date du dernier commit")

        # Écoute uniquement les changements de filtre Git
        self._update()

    @param.depends('filter_model.df_git', watch=True)
    def _update(self, *events):
        df_filtered = self.filter_model.df_git
        self.commit_count.value = len(df_filtered)
        df_commands = pn.state.cache['db']['command']
        df_exec = pn.state.cache['db']['exec']
        if not df_filtered.empty:
            self.git_version_count.value = df_exec[df_exec['sha1'].isin(self.filter_model.get_sha1_valids())]['sha1'].nunique()

            latest_date = df_filtered['date'].max()
            self.last_commit_text.value = latest_date.strftime('%Y-%m-%d %H:%M:%S')
        else:
            self.git_version_count.value = 0
            self.last_commit_text.value = "Aucune date disponible"

    def __panel__(self):
        return pn.Row(self.commit_count, self.git_version_count, self.last_commit_text)


class RunsCountIndicator(pn.viewable.Viewer):
    git_filter = param.ClassSelector(class_=GitFilterModel)

    def __init__(self, **params):
        super().__init__(**params)

        # Initialisation du widget
        self.indicator = pn.indicators.Number(
            name="Exécutions par pas de SNR chargés en base",
            value=self._compute_value(),
        )

        # Met à jour automatiquement si le modèle change
        self.git_filter.param.watch(self._update_value, 'date_range')

    def _compute_value(self):
        db = pn.state.cache.get('db')
        return db['runs'].shape[0] if not db['runs'].empty else 0

    def _update_value(self, event=None):
        self.indicator.value = self._compute_value()

    def __panel__(self):
        return self.indicator

##################################### Niveau 2 : Commandes ####################################

#################################
## Sélecteur de configuration ###
#################################

MAX_SELECTION = 10

class Lvl2_ConfigPanel(pn.viewable.Viewer):
    lv2_model = param.ClassSelector(class_=Lvl2_Filter_Model)

    def __init__(self, **params):
        super().__init__(**params)
        self.config_selector = pn.widgets.MultiChoice(name="Sélectionnez les configurations", options=[], sizing_mode="stretch_width")
        self.clear_button = pn.widgets.Button(name="Reset", button_type="warning", width = BUTON_WIDTH)
        self.dialog = pn.pane.Alert(alert_type="danger", visible=False, sizing_mode="stretch_width")

        self.clear_button.on_click(self.clear_configs)

        self.config_selector.param.watch(self._check_selection_limit, 'value')
        self._update_options()
        
    def __panel__(self):
        return pn.Row(
            self.config_selector,
            self.clear_button,
            self.dialog
        )

    @param.depends('lv2_model.options', watch=True)
    def _update_options(self, *events):
        options = self.lv2_model.options["Config_Alias"].tolist()
        self.config_selector.options = options

    def _check_selection_limit(self, event):
        selected = event.new
        if len(selected) > MAX_SELECTION:
            self.config_selector.value = event.old
            self.dialog.open(f"❌ Maximum {MAX_SELECTION} configurations.")
        else:
            # Met à jour la liste des commandes sélectionnées dans le modèle en recupérant l'index pour l'alias
            df = pn.state.cache['db']['command']
            self.lv2_model.value_commands = df[df["Config_Alias"].isin(selected)].index.tolist()

    def clear_configs(self, event=None):
        self.config_selector.value = []

class Lvl2_Git_Selector(pn.viewable.Viewer):
    lv2_model = param.ClassSelector(class_=Lvl2_Filter_Model)
    
    def __init__(self, **params):
        super().__init__(**params)
        self.git_selector = pn.widgets.MultiChoice(name="Sélectionnez les commits", options=[], sizing_mode="stretch_width")
        self.select_all_button = pn.widgets.Button(name="Tout", button_type="success" , width = BUTON_WIDTH)
        self.clear_button = pn.widgets.Button(name="Reset", button_type="warning", width = BUTON_WIDTH)
        self.dialog = pn.pane.Alert(alert_type="danger", visible=False, sizing_mode="stretch_width")

        self.select_all_button.on_click(self.select_all_sha1)
        self.clear_button.on_click(self.clear_sha1)

        self.git_selector.param.watch(self._check_selection_limit, 'value')

        self._update_options()
        
    def __panel__(self):
        return pn.Row(
            self.git_selector,
            pn.Column(
                self.select_all_button,
                self.clear_button
            ),
            self.dialog
        )

    @param.depends('lv2_model.options', watch=True)
    def _update_options(self, *events):
        options = self.lv2_model.options["sha1"].unique().tolist()
        self.git_selector.options = options
        self.select_all_button.disabled = len(options) > MAX_SELECTION

    def _check_selection_limit(self, event):
        selected = event.new
        if len(selected) > MAX_SELECTION:
            self.git_selector.value = event.old
            self.dialog.open(f"❌ Maximum {MAX_SELECTION} configurations.")
        else:
            self.lv2_model.value_sha1 = selected


    def select_all_sha1(self, event=None):
        if len(self.git_selector.options) > MAX_SELECTION:
            self.dialog.open(f"⚠️ Plus de {MAX_SELECTION} configurations. Filtrez avant de tout sélectionner.")
        else:
            self.git_selector.value = self.git_selector.options

    def clear_sha1(self, event=None):
        self.git_selector.value = []
  
class TableConfig(pn.viewable.Viewer):
    lvl2_model = param.ClassSelector(class_=Lvl2_Filter_Model)
    
    def __init__(self, **params):
        super().__init__(**params)
        self.tab =  pn.pane.DataFrame(self._prepare(), name='table.selected_config', index=True)

    def __panel__(self):
        return pn.Accordion( ("Configurations sélectionnées", self.tab))
    
    @param.depends('lvl2_model.value_commands', 'lvl2_model.value_sha1', watch=True)
    def _update_table(self, event=None):
        self.tab.object = self._prepare()

    def _prepare(self):
        db = pn.state.cache['db']
        df_filtered = self.lvl2_model.df_commands[['param_id']].merge(
            db['parameters'], 
            left_on='param_id', 
            right_index=True
        ).drop(columns=['param_id'])
        return df_filtered

##################################### Niveau 3 : Commande ####################################

# ------------------------------------------------------------------
# Filtres des données niveau 3
# ------------------------------------------------------------------

class Level3(pn.viewable.Viewer):
    lvl2_model = param.ClassSelector(class_=Lvl2_Filter_Model)

    def __init__(self, **params):
        super().__init__(**params)

        self.unique_model = ConfigUniqueModel(lv2_model=self.lvl2_model)

        self.selector_command = ConfigUniqueSelector(name="One Configuration Selection", model=self.unique_model)

        self.pan1 = ExecutionColumn(
            exec_model=ExecUniqueModel(unique_conf_model=self.unique_model),
            name="Execution 1",
            noise_scale=noiseScale
        )

        self.pan2 = ExecutionColumn(
            exec_model=ExecUniqueModel(unique_conf_model=self.unique_model),
            name="Execution 2",
            noise_scale=noiseScale
        )

        self.warning = pn.pane.Markdown("Aucune configuration sélectionnée ou aucune donnée.")
        self.warning.visible = False  # initialement caché

        self.main_panel = pn.Column(
            self.selector_command,
            pn.Row(self.pan1, self.pan2, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
            visible=False  # initialement caché
        )

        self.view = pn.Column(self.warning, self.main_panel)

        # Mise à jour initiale
        self._update_visibility()

        # Lier la mise à jour à des changements
        self.lvl2_model.param.watch(self._update_visibility, ['value_commands', 'value_sha1'])

    def _update_visibility(self, *_):
        visible = bool(self.unique_model.options)

        self.main_panel.visible = visible
        self.warning.visible = not visible

    def __panel__(self):
        return self.view


class ConfigUniqueSelector(pn.viewable.Viewer):
    model = param.ClassSelector(class_=ConfigUniqueModel)

    def __init__(self, **params):
        super().__init__(**params)

        # RadioBoxGroup initialisé avec les alias disponibles
        self.selector = pn.widgets.RadioBoxGroup(
            name='Commandes',
            options=self.model.options_alias,
            value=self.model.alias(),
            inline=False
        )

        # Lorsque l'utilisateur change la sélection, on met à jour self.model.value
        self.selector.param.watch(self._sync_model_from_selector, 'value')
        
    def _sync_model_from_selector(self, event):
        """Binde la sélection (alias) vers le model.value."""
        if event.new:
            self.model.config = command_alias_to_command_id(event.new)

    @param.depends('model.config', 'model.options', watch=True)
    def _sync_selector_from_model(self, event=None):
        self.selector.options  = self.model.options_alias
        self.selector.value = self.model.alias()
        if self.selector.value is None:
            self.selector.value = self.model.options_alias[0] if self.model.options_alias else None
        
    def __panel__(self):
        return pn.Column(
            pn.pane.Markdown("**Choisir une commandes :**"),
            self.selector
        )

class ExecUniqueSelector(pn.viewable.Viewer):
    execUniqueModel = param.ClassSelector(default=None, class_=ExecUniqueModel)

    def __init__(self, **params):
        super().__init__(**params)

        self._syncing = False  # évite les boucles de synchronisation

        self.exec_selector = pn.widgets.Select(
            name="Choisir une exécution",
            options=[],
        )
        self._sync_selector_from_model()
        self.exec_selector.param.watch(self._sync_model_from_selector, "value")

    @param.depends('execUniqueModel.log_hash', 'execUniqueModel.trig_opt_changed', watch=True)
    def _sync_selector_from_model(self):
        """Synchronise le widget avec le modèle."""
        if self._syncing:
            return

        self._syncing = True
        try:
            label_map = self.execUniqueModel.options
            self.exec_selector.options = self.execUniqueModel.options
            self.exec_selector.value = exec_log_hash_to_alias(self.execUniqueModel.log_hash) if self.execUniqueModel.log_hash else None

        finally:
            self._syncing = False

    def _sync_model_from_selector(self, event):
        """Met à jour le modèle à partir de la sélection utilisateur."""
        if self._syncing or self.execUniqueModel is None:
            return

        self._syncing = True
        try:
            selected_label = event.new
            self.execUniqueModel.log_hash = exec_alias_to_log_hash(selected_label)
        finally:
            self._syncing = False

    def __panel__(self):
        return self.exec_selector

# ------------------------------------------------------------------
# Affichage d'une execution
# ------------------------------------------------------------------
class ExecutionColumn(pn.viewable.Viewer):
    def __init__(self, name, exec_model, noise_scale, **params):
        super().__init__(**params)

        # Sélecteur d'exécution
        self.exec_selector = ExecUniqueSelector(name=name, execUniqueModel=exec_model)

        # Histogramme des tâches
        self.histogram = Tasks_Histogramme(
            unique_exec_model=exec_model,
            noiseScale=noise_scale
        )

        # Visualiseur de logs
        self.log_viewer = LogViewer(execUniqueModel=exec_model)

        self.layout = pn.Column(
            self.exec_selector,
            self.histogram,
            pn.pane.HTML(f"<h3> ✏️ Logs - {name}</h3>"),
            self.log_viewer,
            sizing_mode="stretch_width"
        )

    def __panel__(self):
        return self.layout
    
# ------------------------------------------------------------------
# Affichage des journeaux d'exec
# ------------------------------------------------------------------
class LogViewer(pn.viewable.Viewer):
    execUniqueModel = param.ClassSelector(default=None, class_=ExecUniqueModel)
    
    def __init__(self, **params):
        super().__init__(**params) 
        self.output_pane = pn.pane.Markdown("Sélectionnez une configuration pour voir les fichiers.")

    @param.depends('execUniqueModel.log_hash', watch=True)
    def _update_log(self, event=None):
        self.output_pane.object = self.execUniqueModel.log

    def __panel__(self):
        # Affichage du sélecteur et des onglets
        return self.output_pane

# ------------------------------------------------------------------
# Graphe des tâches
# ------------------------------------------------------------------
class Tasks_Histogramme(pn.viewable.Viewer):
    # Paramètres configurables
    unique_exec_model = param.ClassSelector(class_=ExecUniqueModel, doc="Selecteur de configurations uniques")
    noiseScale = param.ClassSelector(class_=NoiseScale, doc="Choix de l'échelle de bruit par passage du label de la colonne")

    def __init__(self, **params):
        super().__init__(**params)
        self.button_time_perc = pn.widgets.Toggle(name='⏱', value=False)
        self.button_time_perc.param.watch(self.changeIcon, 'value')
        self.ListBouton = pn.Column(
            pn.widgets.TooltipIcon(value="Affichage des temps des tâches en milli-seconde ou en %."), 
            self.button_time_perc,
            width=50)
        self.graphPanel = pn.bind(self._plot_task_data, self.button_time_perc, self.unique_exec_model.param.log_hash, self.noiseScale.param.value)
        
    def changeIcon(self, event) :
        if event.new : 
            self.button_time_perc.name = '%'
        else :
            self.button_time_perc.name = '⏱'
    
    def __panel__(self):
        return pn.Row(self.ListBouton, self.graphPanel)
    
    def _plot_task_data(self, percent, index, noiseKey):
        if index is None :
            return pn.pane.Markdown(f"Histogramme des tâches : Sélectionner une configuration pour afficher.")
        df = self.unique_exec_model.df_tasks

        if df.empty:
            self.button_time_perc.disabled = True
            return pn.pane.Markdown(f"Histogramme des tâches : Pas de données de tâches disponibles pour l'execution de la configuration sélectionnée.")
        else:
            self.button_time_perc.disabled = False
            
        if percent:
            y_label = ('Time', 'Durée')
        else:
            y_label = ('Perc', 'Durée (%)')
        
        # Pivot des données pour que chaque combinaison Signal Noise Ratio(SNR).Eb/N0(dB) ait des colonnes pour les temps des tâches
        pivot_df = df.pivot_table(
            values=y_label[0], 
            index=[noiseKey], 
            columns='Task',
            aggfunc='sum', 
            fill_value=0
        )

        # Générer une palette de couleurs automatiquement selon le nombre de configurations
        colors = px.colors.qualitative.Plotly[:len(index) * len(df['Task'].unique())]

        # Initialiser la figure Plotly
        fig = go.Figure()
        
        # Ajouter chaque tâche comme une barre empilée
        for task in pivot_df.columns:
            fig.add_trace(go.Bar(
                x=pivot_df.index.map(lambda x: f"SNR: {x}"),  # SNR comme étiquette
                y=pivot_df[task],
                name=task
            ))

        # Configuration de la mise en page
        fig.update_layout(
            barmode='stack',
            title=f"Temps des tâches par Configuration et Niveau de Bruit  : {noiseKey}",
            xaxis_title="Niveau de Bruit",
            yaxis_title=y_label[1],
            xaxis=dict(tickangle=25),  # Rotation des étiquettes de l'axe x
            template="plotly_white",
            height=900,
            showlegend=True,
            margin=dict(t=70, b=50, l=50, r=10)
        )
        return pn.pane.Plotly(fig, sizing_mode="stretch_width")

# ------------------------------------------------------------------
# Performance BER/FER par niveau de bruit (lvl2)
# ------------------------------------------------------------------
class PerformanceBERFERPlot(pn.viewable.Viewer):
    lvl2_model = param.ClassSelector(class_=Lvl2_Filter_Model, doc="Modèle de filtrage de niveau 2")
    noise_scale_param = param.ClassSelector(class_=NoiseScale, doc="Paramètre de niveau de bruit")
    
    def __init__(self, **params):
        super().__init__(**params)
        self.plot_pane = pn.pane.Plotly(sizing_mode="stretch_width")
        self.view = self.plot_pane

        # Initialisation + lien des callbacks
        self._update_plot()
        self.lvl2_model.param.watch(self._update_plot, ['value_commands', 'value_sha1'])
        self.noise_scale_param.param.watch(self._update_plot, ['value'])

    def __panel__(self):
        return self.view

    def _update_plot(self, *_):
        df_runs = self.lvl2_model.df_runs
        
        if df_runs.empty:

            fig = go.Figure()
            fig.add_annotation(
                text="⚠️ Aucune exécution sélectionnée ou disponible.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20),
                align="center"
            )
            fig.update_layout(
                template="plotly_white",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=600
            )
            self.plot_pane.object = fig
            return

        noiseScale = self.noise_scale_param.value
        df_runs = df_runs.sort_values(by=noiseScale, ascending=True)

        # Ajouter la colonne clé (couple Command_id + sha1)
        df_runs['cmd_sha'] = (
            df_runs['Command_id'].astype(str) + ' - ' +
            df_runs['sha1'].str[:7]
        )

        grouped = df_runs.groupby('cmd_sha')
        colors = px.colors.qualitative.Plotly[:len(grouped)]

        fig = go.Figure()

        for (key, grp), color in zip(grouped, colors):
            snr = grp[noiseScale]
            ber = grp['Bit Error Rate (BER) and Frame Error Rate (FER).BER']
            fer = grp['Bit Error Rate (BER) and Frame Error Rate (FER).FER']

            # Trace BER (ligne pleine avec marqueurs)
            fig.add_trace(go.Scatter(
                x=snr, y=ber,
                mode='lines+markers',
                name=f"BER - {key}",
                line=dict(width=2, color=color),
                marker=dict(symbol='circle', size=6)
            ))

            # Trace FER (ligne pointillée avec marqueurs)
            fig.add_trace(go.Scatter(
                x=snr, y=fer,
                mode='lines+markers',
                name=f"FER - {key}",
                line=dict(width=2, dash='dash', color=color),
                marker=dict(symbol='x', size=6)
            ))

        
        # Configuration de la mise en page avec Range Slider et Range Selector
        fig.update_layout(
            title="BER et FER en fonction du SNR pour chaque couple (Command, Commit)",
            xaxis=dict(
                title=f"Niveau de Bruit (SNR) : {noiseScale}",
                rangeslider=dict(visible=True),
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1dB", step="all", stepmode="backward"),
                        dict(count=5, label="5dB", step="all", stepmode="backward"),
                        dict(count=10, label="10dB", step="all", stepmode="backward"),
                        dict(step="all")
                    ]
                )
            ),
            yaxis=dict(title="Taux d'Erreur", type='log'),
            legend_title="Command - Commit",
            template="plotly_white",
            height=600,
            showlegend=True,
            margin=dict(t=70, b=50, l=50, r=10)
        )

        self.plot_pane.object = fig


# ------------------------------------------------------------------
# Assemblage du panel git
# ------------------------------------------------------------------
class PanelCommit(pn.viewable.Viewer):
    
    command_filter = param.ClassSelector(default=None, class_=CommandFilterModel, doc="Filtre de commandes")
    git_filter = param.ClassSelector(default=None, class_=GitFilterModel, doc="Filtre Git")
    
    def __init__(self, **params):
        super().__init__(**params)
        # Initialisation du tableau de commandes
        self.date_slider    = DateRangeFilter(git_filter=self.git_filter)
        # Composants construits
        self.code_selector = CodeSelector(cmd_filter_model=self.command_filter)
        self.table = FilteredTable(filter_model=self.git_filter)
        db = pn.state.cache['db']
        self.indicators = GitIndicators(filter_model=self.git_filter)
        self.perfgraph = PerformanceByCommit(command_filter=self.command_filter)

    def __panel__(self):
        return pn.Column(
            self.indicators,
            self.date_slider,
            self.table,
            self.code_selector,
            self.perfgraph,
            sizing_mode="stretch_width"
        )

    def update_command_table(self, event=None):
        self.command_table.value = self.command_filter.df_git

# ------------------------------------------------------------------
# Paramêtre du site
# ------------------------------------------------------------------
noiseScale = NoiseScale(noise_label= noise_label)

# ------------------------------------------------------------------
# Point d’entrée unique
# ------------------------------------------------------------------

def main():
    print(ud.unidata_version)
    load_data_sync()
    for k, v in pn.state.cache['db'].items():
        print(f"{k:8s} : {len(v):6d} lignes")
    template = init_dashboard()
    template.servable()

if IS_PANEL_CONVERT:
    # GitHub-Pages (pyodide-worker) → on charge et on sert
    main()

elif IS_PYODIDE:
    # JupyterLite ou autre environnement Pyodide → onload
    pn.state.onload(lambda: main())

else:
    # Mode local « python dashboard_commit.py »
    load_data_sync()
    for k, v in pn.state.cache['db'].items():
        print(f"{k:8s} : {len(v):6d} lignes")
    dashboard = init_dashboard()
    dashboard.show(port=35489)