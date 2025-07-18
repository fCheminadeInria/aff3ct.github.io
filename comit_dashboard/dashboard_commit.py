from bokeh.settings import settings
settings.resources = 'inline'

import pandas as pd
import panel as pn
from datetime import datetime
import argparse
import re
try:
    from pyodide.http import open_url
    from pyodide.http import pyfetch
except ModuleNotFoundError:
    # Utiliser un fallback avec urllib en environnement standard
    from urllib.request import urlopen as open_url
    pyfetch = None
    
import plotly.graph_objs as go
import plotly.express as px
import plotly.graph_objects as go
import param
from panel.viewable import Viewer
import unicodedata as ud
import itertools
from io import BytesIO
import sys
import asyncio
import os

##################################### Niveau Global ####################################

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

pn.config.raw_css.append("""
.align-right {
    margin-left: auto;
    display: flex;
    justify-content: flex-end;
}
""")

pn.config.raw_css = [
    """
    .tittle_indicator-text {
        font-size: 20px;
        font-weight: normal;
        color: #333333;
    }
    .indicator-text {
        font-size: 64px;
        font-weight: normal;
        color: #333333;
    }
    """
]

ACCENT = "teal"

styles = {
    "box-shadow": "rgba(50, 50, 93, 0.25) 0px 6px 12px -2px, rgba(0, 0, 0, 0.3) 0px 3px 7px -3px",
    "border-radius": "4px",
    "padding": "10px",
  }

#logo = pn.pane.Image("/home/fchemina/aff3ct.github.io/comit_dashboard/image/2ada77c3-f7d7-40ae-8769-b77cc3791e84.webp")
#logo = pn.pane.Image("https://raw.githubusercontent.com/fCheminadeInria/aff3ct.github.io/refs/heads/master/comit_dashboard/image/2ada77c3-f7d7-40ae-8769-b77cc3791e84.webp")
logo = pn.pane.Image("https://raw.githubusercontent.com/fCheminadeInria/aff3ct.github.io/refs/heads/master/comit_dashboard/image/93988066-1f77-4b42-941f-1d5ef89ddca2.webp")


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
    df_git = param.DataFrame()
    date_range = param.Tuple(default=(None, None), length=2, doc="Plage de dates pour filtrer")
    filtered = param.Parameter()

    @param.depends('date_range', watch=True)
    def _trigger(self):
        self.param.trigger('filtered')

    def __init__(self, **params):
        super().__init__(**params)
        # Si date_range n'est pas fourni, on initialise avec la plage complète des dates
        if self.date_range == (None, None):
            min_date = self.df_git['date'].min()
            max_date = self.df_git['date'].max()
            self.date_range = (min_date, max_date)


    def get_filtered_df(self):
        df = self.df_git.copy()
        start, end = self.date_range
        if start and end:
            start = start
            end   = end
            df = df[(df['date'] >= start) & (df['date'] <= end)]
        return df
    
    def get_sha1_valids(self):
        return self.get_filtered_df().index.unique()

class CommandFilterModel(param.Parameterized):
    df_commands = param.DataFrame()
    git_filter = param.ClassSelector(class_=GitFilterModel) 
    code = param.ListSelector(default=[], objects=[])
    filtered = param.Parameter() # variable pour déclencher le filtrage
    config_filter = param.Dict(default={})

    def __init__(self, **params):
        super().__init__(**params)
        # Initialisation de 'code' avec toutes les valeurs possibles dans df_commands['code']
        all_codes = sorted(self.df_commands['code'].dropna().unique().tolist())
        self.param['code'].objects = all_codes
        self.param['code'].default = all_codes 
        self.code = all_codes

    @param.depends('config_filter', 'code', 'git_filter.filtered', watch=True)
    def _trigger(self):
        self.param.trigger('filtered')  


    def get_filtered_df(self):
        sha1_valids = self.git_filter.get_sha1_valids()
        df_filtered = self.df_commands[self.df_commands['sha1'].isin(sha1_valids)]
        if 'All' not in self.code:
            df_filtered = df_filtered[df_filtered['code'].isin(self.code)]
        self.df_commands_intermediare = df_filtered
        for col, values in self.config_filter.items():
            if values:
                df_filtered = df_filtered[df_filtered[col].isin(values)]
        return df_filtered


class Research_config_filter(pn.viewable.Viewer):
    command_filter = param.ClassSelector(class_=CommandFilterModel)

    def __init__(self, **params):
        super().__init__(**params)

        df = self.command_filter.get_filtered_df()
        df_filtered = df.drop(columns=['Config_Alias', 'param_id', 'meta_id', 'git_id', 'Command', 'Simulation.Code type (C)'], errors='ignore')

        # Identification des familles de paramètres
        family_columns = {}
        for col in df_filtered.columns:
            match = re.match(r"(\w+)\.(\w+)", col)
            if match:
                family_columns.setdefault(match.group(1), []).append(col)
            else:
                family_columns.setdefault("Autres", []).append(col)

        # Création des widgets de filtrage
        family_widgets = {}
        for family, columns in family_columns.items():
            widgets = []
            for col in columns:
                options = sorted(df[col].dropna().unique().tolist())
                is_disabled = len(options) <= 1
                widget = pn.widgets.MultiChoice(
                    name=col,
                    options=options,
                    value=options,
                    disabled=is_disabled,
                    css_classes=["grayed-out"] if is_disabled else []
                )
                widget.param.watch(self._update_filterconfig, 'value')
                widgets.append(widget)
            family_widgets[family] = pn.Column(*widgets, name=family)

        self.accordion_families = pn.Accordion(*[(f"{family}", widget) for family, widget in family_widgets.items()])
        
        self.filtre_actif= pn.pane.Markdown("", height=100)
        self._config_filter_to_markdown()
        
        self.command_filter.param.watch(self._update_filter, 'filtered')

    def _config_filter_to_markdown(self) -> str:
        parts = []

        for col_container in self.accordion_families.objects:
            for widget in col_container.objects:
                all_options = widget.options
                selected = widget.value
                deselected = sorted(set(all_options) - set(selected))
                if deselected:
                    parts.append(f"**{widget.name}** : {', '.join(map(str, deselected))}")

        self.filtre_actif.object =  "\n\n".join(parts) if parts else "_Aucun filtre désactivé_"
        
    def __panel__(self):
        
        return pn.Card(
            pn.Column(
                pn.Card(
                    pn.Column(
                        self.filtre_actif, 
                        styles={'overflow-y': 'auto'}
                        ),
                    title="🔍 Filtres actifs"
                    ),
                self.accordion_families, 
                height=400,
                styles={'overflow-y': 'auto'}),
                title="🔍 Filtres de recherche",
                collapsed=True
                )
            
    def _get_current_filter(self):
        """Construit un dictionnaire {colonne: valeurs sélectionnées} pour le filtre."""
        return {
            widget.name: widget.value
            for col in self.accordion_families.objects
            for widget in col.objects
            if widget.value  # Ne garde que les filtres actifs
        }

    def _update_filter(self, event):
        df = self.command_filter.get_filtered_df()
        if df is None or df.empty:
            return

        # Appliquer le filtre actuel sur df_commands pour avoir le df filtré
        df_filtered = df.drop(columns=['Config_Alias', 'param_id', 'meta_id', 'git_id', 'Command', 'Simulation.Code type (C)'], errors='ignore')

        # Appliquer le filtre config_filter (ex: garder uniquement les valeurs sélectionnées pour chaque colonne)
        for col, selected_values in self.command_filter.config_filter.items():
            if selected_values:
                df_filtered = df_filtered[df_filtered[col].isin(selected_values)]

        # Met à jour les options et éventuellement les valeurs des widgets
        for col_container in self.accordion_families.objects:
            for widget in col_container.objects:
                if widget.name in df_filtered.columns:
                    options = sorted(self.command_filter.df_commands_intermediare[widget.name].dropna().unique().tolist())
                    widget.options = options

                    # Si la sélection actuelle n'est plus dans les options, on remet toute la sélection possible
                    if not set(widget.value).issubset(set(options)):
                        widget.value = options if options else []

    def _update_filterconfig(self, event):
        """Met à jour le filtre du modèle lors d’un changement utilisateur."""
        
            # Empêche la suppression complète des options
        if len(event.new) < 1:
            event.obj.value = event.old
            return

        self.command_filter.config_filter = {**self.command_filter.config_filter,    event.obj.name: event.new}
        self._config_filter_to_markdown()


################################################
## Gestion des données niveau 2 avec filtrage ##
################################################

class Lvl2_Filter_Model(param.Parameterized):
    command_filter = param.ClassSelector(class_=CommandFilterModel)
    value = param.List(default=[])
    df = param.DataFrame()
    options = param.DataFrame(default=pd.DataFrame(columns=["Config_Alias"]), doc="DataFrame contenant les options de filtrage")

    def __init__(self, **params):
        super().__init__(**params)
        self._update_df()
        self._update_from_lvl1()
        self.command_filter.param.watch(self._update_from_lvl1, 'filtered')
        self.param.watch(self._update_df, 'value')
        
    
    def _update_from_lvl1(self, *events):
        df_filtered = self.command_filter.get_filtered_df()
        self.value = [v for v in self.value if v in df_filtered.index]
        self.options = df_filtered[['Config_Alias']]

    @property
    def df_runs_filtred(self):
        df_runs = pn.state.cache['db']['runs']
        df_runs = df_runs[df_runs["Command_id"].isin(self.value)]
        return  df_runs
            
    def _update_df(self, *events):
        self.df = self.command_filter.get_filtered_df().loc[self.value]     
     
    def reset(self):
        self.value = []


##################################################
## Gestion des données niveau 3 : config unique ##
##################################################

class ConfigUniqueModel(param.Parameterized):
    lv2_model = param.ClassSelector(default=None, class_=Lvl2_Filter_Model)
    value = param.Selector(default=None, objects=[])
    date = param.Selector(default=None, objects=[])
    options = param.Selector(default=None, objects=[])


    @property
    def _df_configs_from_lvl2(self):
        """Accès sécurisé au DataFrame."""
        return self.lv2_model.df if self.lv2_model is not None else pd.DataFrame()

    @property
    def df(self):
        if self.value is None:
            return self._df_configs_from_lvl2.iloc[0:0]  # DataFrame vide
        return self._df_configs_from_lvl2.loc[self.value]

    @property
    def df_runs(self):
        db = pn.state.cache.get('db', {})
        if 'runs' not in db or self.value is None:
            return pd.DataFrame()
        return  db['runs'][db['runs']['Command_id']== self.value]        

    @property
    def df_logs(self):
        db = pn.state.cache.get('db', {})
        if 'logs' not in db or self.value is None:
            return pd.DataFrame()
        df_logs = db['logs']
        log_hash = self.df_runs['log_hash'].unique() if not self.df_runs.empty else []
        return  df_logs[df_logs['hash'].isin(log_hash)]

    @property
    def log(self):
        df_logs = self.df_logs
        if self.date is None or df_logs.empty:
            return "```Pas de logs pour la sélection.```"
        
        match = df_logs[df_logs['Date_Execution'] == self.date]
        if match.empty:
            return "```Aucun log trouvé pour cette date.```"
        
        if 'log' in match.columns:
            # Encapsuler le log dans un bloc de code Markdown
            contenu = match.iloc[0]['log']
            return f"```\n{contenu}\n```"
        else:
            return "```Colonne 'log' manquante.```"

    @param.depends('value', watch=True)
    def _update_date(self):
        if 'hash' in self.df_logs.columns :
            self.date = self.df_logs['Date_Execution'].iloc[0]
        else:
            self.date = None
            
    @property
    def options_dates(self):
        """Liste des dates d'exécution disponibles pour la configuration sélectionnée."""
        df_logs = self.df_logs
        if df_logs.empty or 'Date_Execution' not in df_logs.columns:
            return []
        # Conversion en str pour l'affichage (utile pour un widget Select)
        return df_logs['Date_Execution'].astype(str).unique().tolist()
 
    @property
    def options_alias(self):
        return self._df_configs_from_lvl2['Config_Alias'].tolist()

    def _find_id_by_alias(self, alias):
        df = self._df_configs_from_lvl2
        if df.empty or 'Config_Alias' not in df.columns:
            return None
        matched = df.index[df['Config_Alias'] == alias]
        return matched[0] if len(matched) > 0 else None

    def alias(self):
        if self.value is None or self.value not in self._df_configs_from_lvl2.index:
            return '-'
        return self._df_configs_from_lvl2.at[self.value, 'Config_Alias']

    def value_by_alias(self, alias):
        id = self._find_id_by_alias(alias)
        if id is not None:
            self.value = id

    
    @param.depends('lv2_model.df', watch=True)
    def _on_lvl2_df_change(self):
        opts = self._df_configs_from_lvl2.index.tolist()
        # Initialise la valeur avec le command_id correspondant au premier alias
        if self.value not in opts :
            self.value = opts[0] if opts else None
        self.options = opts


##################################### Niveau 1 : Git et perf global ####################################


#################################
## Component pour le Panel Git ##
#################################
class DateRangeFilter(pn.viewable.Viewer):
    git_filter = param.ClassSelector(class_=GitFilterModel)

    def __init__(self, **params):
        super().__init__(**params)
        # Bornes extraites du DataFrame Git
        db = pn.state.cache['db']
        df = db ['git']
        
        start, end = df['date'].min(), df['date'].max()
        
        # Forcer la plage de dates dans le modèle si elle est incorrecte ou absente
        if not hasattr(self.git_filter, 'date_range') or self.git_filter.date_range is None:
            self.git_filter.date_range = (start, end)
        
        # Création du slider
        self.slider = pn.widgets.DatetimeRangeSlider(
            name='Filtre sur les dates des commits',
            start=start,
            end=end,
            value=(start, end),
            sizing_mode='stretch_width',
            step = 300,
        )
        self.slider.param.watch( lambda event: setattr(self.git_filter, 'date_range', event.new),'value')

    def __panel__(self):
        return self.slider

class PerformanceByCommit(pn.viewable.Viewer):
    git_filter = param.ClassSelector(class_=GitFilterModel)
    command_filter = param.ClassSelector(class_=CommandFilterModel)
    
    def __init__(self, **params):
        super().__init__(**params)

        self.git_filter.param.watch(self._update_all, 'filtered')
        self.command_filter.param.watch(self._update_all, 'filtered')
        
        self.plot_throughput_pane = pn.pane.Plotly(sizing_mode='stretch_width')
        self.plot_latency_pane = pn.pane.Plotly(sizing_mode='stretch_width')
        
        df_commands = self.command_filter.df_commands
        
        db = pn.state.cache['db']
        
        # filtrage sur les commandes restantes et ajouts des colonnes de date
        df = db['runs'][db['runs']['Command_id'].isin(df_commands.index)].merge(
           df_commands[['sha1', 'code']], left_on='Command_id', right_index=True
        ).merge(
            db['git'][['date']], left_on='sha1', right_index=True
        ).reset_index(drop=True)
        
        self.df = df.sort_values(by=['date'])
        
        self._update_all()
        
        self.tabs = pn.Tabs(
            ('⏱️ Latence', self.plot_latency_pane),
            ('📈 Débit', self.plot_throughput_pane),
        )
        
    def _update_all(self, *events):
        self._update_data()
        self._create_plots()
        self.plot_throughput_pane.object = self.fig_throughput
        self.plot_latency_pane.object = self.fig_latency

    def _update_data(self):

        df = self.df[
            (self.df['sha1'].isin(self.git_filter.get_filtered_df().index)) &
            (self.df['Command_id'].isin(self.command_filter.get_filtered_df().index))
        ]
        
        # Aggrégation des données par commit et par type de code
        throughput_col = 'Global throughputand elapsed time.SIM_THR(Mb/s)'
        latency_col = 'Global throughputand elapsed time.elapse_time(ns)'
        
        self.df_grouped = df.groupby(['sha1', 'code']).agg({
            throughput_col: 'mean',
            latency_col: 'mean',
            'date': 'first'
        }).reset_index().rename(columns={
            throughput_col: 'Débit moyen (Mb/s)',
            latency_col: 'Latence moyenne (ns)',
            'code' : 'Code',
        }).sort_values(by=['date'])

    def _create_plots(self):
        self.fig_throughput = px.line(
            self.df_grouped,
            x='date', y='Débit moyen (Mb/s)',
            color='Code',
            title="Débit moyen par commit (par code)",
            markers=True
        )
        self.fig_throughput.update_layout(
            legend=dict(orientation='v', y=1, x=1.05),
            margin=dict(r=100),
            xaxis=dict(title="Date", rangeslider=dict(visible=True), showgrid=True),
            yaxis=dict(title="Débit moyen (Mb/s)", showgrid=True),
        )

        self.fig_latency = px.line(
            self.df_grouped,
            x='date', y='Latence moyenne (ns)',
            color='Code',
            title="Latence moyenne par commit (par code)",
            markers=True
        )
        self.fig_latency.update_layout(
            legend=dict(orientation='v', y=1, x=1.05),
            margin=dict(r=100),
            xaxis=dict(title="Date", rangeslider=dict(visible=True), showgrid=True),
            yaxis=dict(title="Latence moyenne (ns)", showgrid=True),
        )

    def __panel__(self):
        return self.tabs

##########################
## Sélecteur de code ##
##########################
class CodeSelector(pn.viewable.Viewer):
    cmd_filter_model = param.ClassSelector(class_=CommandFilterModel)

    def __init__(self, **params):
        super().__init__(**params)
        self.widget = pn.widgets.CheckBoxGroup(name='Codes à afficher', inline=True)
        
        db = pn.state.cache['db']
        self.widget.options = sorted(db['param']['Simulation.Code type (C)'].fillna('Non défini').unique().tolist())
        self.cmd_filter_model.param['code'].objects = self.widget.options   
        self.widget.value = self.cmd_filter_model.param['code'].default  # Affecte la valeur par défaut des codes     
        # self.widget.param.watch(self._update_filter, 'value')
        
        self.select_all_button = pn.widgets.Button(name='Sélectionner tout', button_type='primary')
        self.select_all_button.on_click(self.select_all_codes)
        
        self.deselect_all_button = pn.widgets.Button(name='Désélectionner tout', button_type='danger')
        self.deselect_all_button.on_click(self.deselect_all_codes)
        
        self.apply_button = pn.widgets.Button(name='Appliquer les filtres', button_type='success') 
        self.apply_button.on_click(self._update_filter) 

        self.spinner = pn.indicators.LoadingSpinner(value=False, width=25)
        
    def select_all_codes(self, event):
        self.widget.value = self.widget.options

    def deselect_all_codes(self, event):
        self.widget.value = []

    def _update_filter(self, event):
        self.cmd_filter_model.code = self.widget.value
        
        self.spinner.value = True

        try:
            self.cmd_filter_model.code = self.widget.value
        finally:
            self._set_interactive(True)
            self.spinner.value = False

    def _set_interactive(self, active: bool):
        """Active ou désactive les interactions"""
        self.widget.disabled = not active
        self.select_all_button.disabled = not active
        self.deselect_all_button.disabled = not active
        self.apply_button.disabled = not active

    def __panel__(self):
        return pn.Row(self.select_all_button, self.deselect_all_button, self.widget, self.apply_button, self.spinner)
    
##############################
## Table des commits Git ##
##############################

class FilteredTable(pn.viewable.Viewer):
    filter_model = param.ClassSelector(class_=GitFilterModel)

    def __init__(self, **params):
        super().__init__(**params)
        self.table = pn.widgets.DataFrame(height=300, text_align='center', sizing_mode="stretch_width")
        self._update()
        self.filter_model.param.watch(self._update, ['filtered'])

    def _update(self, *events):
        self.table.value = self.filter_model.get_filtered_df()

    def __panel__(self):
        return self.table

#####################
## Indicateurs Git ##
#####################

class GitIndicators(pn.viewable.Viewer):
    filter_model = param.ClassSelector(class_=GitFilterModel)

    def __init__(self, df_git, df_commands, **params):
        super().__init__(**params)
        self.df_git = df_git
        self.df_commands = df_commands

        self.commit_count = pn.indicators.Number(name="Commits historisés dans Git", value=0)
        self.git_version_count = pn.indicators.Number(name="Commits avec des données", value=0)
        self.last_commit_text = pn.widgets.StaticText(name="Date du dernier commit")

        # Écoute uniquement les changements de filtre Git
        self.filter_model.param.watch(self._update, 'filtered')
        self._update()

    def _update(self, *events):
        df_filtered = self.filter_model.get_filtered_df()
        self.commit_count.value = len(df_filtered)

        if not df_filtered.empty:
            valid_sha1 = df_filtered.index
            count_valid_sha1 = self.df_commands[self.df_commands['sha1'].isin(valid_sha1)]['sha1'].nunique()
            self.git_version_count.value = count_valid_sha1

            latest_date = df_filtered['date'].max()
            self.last_commit_text.value = latest_date.strftime('%Y-%m-%d %H:%M:%S')
        else:
            self.git_version_count.value = 0
            self.last_commit_text.value = "Aucune date disponible"

    def __panel__(self):
        return pn.Row(self.commit_count, self.git_version_count, self.last_commit_text)


##################################### Niveau 2 : Commandes ####################################

#################################
## Sélecteur de configuration ###
#################################

MAX_SELECTION = 10

class ConfigPanel(pn.viewable.Viewer):
    lv2_model = param.ClassSelector(class_=Lvl2_Filter_Model)

    def __init__(self, **params):
        super().__init__(**params)

        self.config_selector = pn.widgets.MultiChoice(name="Sélectionnez les configurations", options=[])
        self.select_all_button = pn.widgets.Button(name="Tout sélectionner", button_type="success")
        self.clear_button = pn.widgets.Button(name="Tout désélectionner", button_type="warning")
        self.dialog = pn.pane.Alert(alert_type="danger", visible=False, sizing_mode="stretch_width")

        self.select_all_button.on_click(self.select_all_configs)
        self.clear_button.on_click(self.clear_configs)

        self.config_selector.param.watch(self._check_selection_limit, 'value')
        self.lv2_model.param.watch(self._update_options, 'options')
        self._update_options()
        
    def __panel__(self):
        return pn.Column(
            self.select_all_button,
            self.clear_button,
            self.config_selector,
            self.dialog
        )

    def _update_options(self, *events):
        options = self.lv2_model.options["Config_Alias"].tolist()
        self.config_selector.options = options
        self.select_all_button.disabled = len(options) > MAX_SELECTION

    def _check_selection_limit(self, event):
        selected = event.new
        if len(selected) > MAX_SELECTION:
            self.config_selector.value = event.old
            self.dialog.open(f"❌ Maximum {MAX_SELECTION} configurations.")
        else:
            self.lv2_model.value = self.lv2_model.options[self.lv2_model.options["Config_Alias"].isin(selected)].index.tolist()

    def select_all_configs(self, event=None):
        if len(self.config_selector.options) > MAX_SELECTION:
            self.dialog.open(f"⚠️ Plus de {MAX_SELECTION} configurations. Filtrez avant de tout sélectionner.")
        else:
            self.config_selector.value = self.config_selector.options

    def clear_configs(self, event=None):
        self.config_selector.value = []
      
# affichage de la sélection     
class TableConfig(pn.viewable.Viewer):
    lv2_filter = param.ClassSelector(class_=Lvl2_Filter_Model)
    meta = param.Boolean(doc="affiche les Meta-données si Vrai, les paramètres de simmulation si faux")
    
    def __init__(self, **params):
        super().__init__(**params)
        self.tab =  pn.pane.DataFrame(self._prepare(), name='table.selected_config', index=True)
        self.lv2_filter.param.watch(self._update_table, 'value')

    def __panel__(self):
        return pn.Accordion( ("📥 Selected Configuration", self.tab))
    
    def _update_table(self, event=None):
        self.tab.object = self._prepare()

    def _prepare(self):
        db = pn.state.cache['db']
        if self.meta :
            df_filtered = self.lv2_filter.df[['meta_id']] .merge(db['meta'] , left_on='meta_id',  right_index=True).drop(columns=['meta_id'])
        else :
            df_filtered = self.lv2_filter.df[['param_id']].merge(db['param'], left_on='param_id', right_index=True).drop(columns=['param_id'])
        
        return df_filtered

class Panel_graph_envelope(pn.viewable.Viewer):
    # Paramètres configurables
    df = param.DataFrame(doc="Le dataframe contenant les données")
    lab = param.String(default="y", doc="Nom de la colonne pour l'axe Y")
    lab_group = param.String(default=None, doc="Nom de la colonne pour regrouper les données")
    labmin = param.String(default=None, doc="Nom de la colonne pour la valeur minimale")
    labmax = param.String(default=None, doc="Nom de la colonne pour la valeur maximale")
    Ytitle = param.String(default="Valeur", doc="Titre de l'axe Y")
    noiseScale = param.ClassSelector(default=None, class_=pn.viewable.Viewer,doc="Choix de l'échelle de bruit par passage du label de la colonne")
    lv2_model = param.ClassSelector(default=None, class_=Lvl2_Filter_Model, doc="Modèle de filtrage de niveau 2")

    def __init__(self, **params):
        super().__init__(**params)
        
        self.button_env = pn.widgets.Toggle(name='〽️', value=True)
        
        if (self.labmin == None or self.labmax == None):
            self.button_env.value=False
            self.button_env.disabled=True
        
        self.ListBouton = pn.Column(
            pn.widgets.TooltipIcon(value="Activer/Désactiver Enveloppe"), 
            self.button_env,
            width=50)
        self.graphPanel = pn.bind(self._plot_enveloppe_incertitude,self.button_env, self.noiseScale.param.value)
        

    def __panel__(self):
        return pn.Row(self.ListBouton, self.graphPanel)
        
    def _plot_enveloppe_incertitude(self, show_envelope, noiseKey):    
        
        index = self.lv2_model.value
        
        if index is None :
            df_filtred = self.df
        else :
            df_filtred = self.df[self.df["Command_id"].isin(index)] 
        
        # Si pas de données de tâches pour les configurations sélectionnées
        if df_filtred.empty:
            self.button_env.disabled=True
            if self.lab_group :
                return pn.pane.Markdown(f"Graphes de {self.Ytitle} : Pas de données de {self.lab_group} disponibles pour les configurations sélectionnées.")
            else :
                return pn.pane.Markdown(f"Graphes de {self.Ytitle} : Pas de données disponibles pour les configurations sélectionnées.")
        else :
            self.button_env.disabled=False
            
        if (self.labmin == None or self.labmax == None):
            show_envelope = False
        
        color_cycle = itertools.cycle(px.colors.qualitative.Plotly)

        fig = go.Figure()

        # Ajouter une trace pour chaque configuration et tâche
        for i, config in enumerate(index):
            # Filtrer les données pour chaque configuration
            config_data = df_filtred[df_filtred['Command_id'] == config]
            
            db = pn.state.cache['db']
            alias = db['commands'].loc[config, 'Config_Alias'] #variable global pas propre mais commode
            if self.lab_group :
                for j, t in enumerate(config_data[self.lab_group].unique()):  
                    task_data = config_data[config_data[self.lab_group] == t]
                    snr = task_data[noiseKey]
                    y_values = task_data[self.lab]         
                    
                    color = next(color_cycle)

                    
                    if show_envelope :
                        y_values_min = task_data[self.labmin]  
                        y_values_max = task_data[self.labmax]   
                        
                        # Courbe pour la latence avec enveloppe
                        fig.add_trace(go.Scatter(
                            x=snr, y=y_values_max,
                            fill=None, mode='lines+markers',
                            line=dict(width=2, dash='dash', color=color),
                            marker=dict(symbol='x', size=6),
                            showlegend=False
                        ))
                        fig.add_trace(go.Scatter(
                            x=snr, y=y_values_min,
                            fill='tonexty', mode='lines+markers',
                            line=dict(width=2, dash='dash', color=color),
                            marker=dict(symbol='x', size=6),
                            name=f"min/max - {alias} - {t}"  
                        ))

                    
                    fig.add_trace(go.Scatter(
                        x=snr, y=y_values,
                        mode='lines+markers',
                        line=dict(width=2, color=color),
                        name=f"{self.lab} - {alias} - {t}"  
                    ))
            else :
                
                color = next(color_cycle)

                snr = config_data[noiseKey]
                y_values = config_data[self.lab]         
                
                if show_envelope :
                    y_values_min = config_data[self.labmin]  
                    y_values_max = config_data[self.labmax]   
                    
                    # Courbe pour la latence avec enveloppe
                    fig.add_trace(go.Scatter(
                        x=snr, y=y_values_max,
                        fill=None, mode='lines+markers',
                        line=dict(width=2, dash='dash', color=color),
                        marker=dict(symbol='x', size=6),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=snr, y=y_values_min,
                        fill='tonexty', mode='lines+markers',
                        line=dict(width=2, dash='dash', color=color),
                        marker=dict(symbol='x', size=6),
                        name=f"min/max - {config}"  
                    ))

                
                fig.add_trace(go.Scatter(
                    x=snr, y=y_values,
                    mode='lines+markers',
                    line=dict(width=2, color=color),
                    name=f"{self.lab} - {config}"  
                ))
        
        # Configuration de la mise en page avec Range Slider et Range Selector
        fig.update_layout(
            title="Latence en fonction du SNR pour chaque configuration",
            xaxis=dict(
                title=f"Niveau de Bruit (SNR) : {noiseKey}",
                rangeslider=dict(visible=True),
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1dB", step="all", stepmode="backward"),
                        dict(count=5, label="5dB", step="all", stepmode="backward"),
                        dict(count=10, label="10dB", step="all", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            ),
            yaxis=dict(
                title=self.Ytitle,
            ),
            legend_title="Configurations",
            template="plotly_white",
            height=600,
            showlegend=True,
            margin=dict(t=70, b=50, l=50, r=10)
        )
        
        return  pn.pane.Plotly(fig, sizing_mode="stretch_width")

class Mutual_information_Panels (pn.viewable.Viewer) :
    # Paramètres configurables
    lv2_model = param.ClassSelector(default=None, class_=Lvl2_Filter_Model)
    index_selecter = param.ClassSelector(default=None, class_=pn.viewable.Viewer, doc="Widget MultiChoice")
    noiseScale = param.ClassSelector(default=None, class_=pn.viewable.Viewer,doc="Choix de l'échelle de bruit par passage du label de la colonne")

    def __init__(self, **params):
        super().__init__(**params)
        
        self.colors = itertools.cycle(px.colors.qualitative.Plotly)
        
        
        df =self.lv2_model.df_runs_filtred
        cols = ["Mutual Information.MI", "Mutual Information.MI_min", "Mutual Information.MI_max", "Mutual Information.n_trials"]
        df = df [ df[cols].notnull().any(axis=1) ]
        
        self.plot_mutual_information = Panel_graph_envelope(
            lv2_model = self.lv2_model,
            df = df,
            lab   ="Mutual Information.MI", 
            labmin="Mutual Information.MI_min", 
            labmax="Mutual Information.MI_max", 
            Ytitle = "Information mutuelle",
            noiseScale = self.noiseScale
        )
        
        self.ListBouton = pn.Column(
            pn.widgets.TooltipIcon(value="Seuls les configuration avec des valeurs pour \"Mutual Information.MI\", \"Mutual Information.MI_min\", \"Mutual Information.MI_max\", \"Mutual Information.n_trials\" sont affichées. "), 
            width=50)
        self.mutual_information_ntrial = pn.bind(self._plottrial, self.lv2_model.param.value, self.noiseScale.param.value)

    def __panel__(self):
        return pn.Column(
            pn.widgets.TooltipIcon(value="Seuls les configuration avec des valeurs pour \"Mutual Information.MI\", \"Mutual Information.MI_min\", \"Mutual Information.MI_max\" sont affichées. "), 
            pn.Row(self.plot_mutual_information),
            pn.Row(self.ListBouton, self.mutual_information_ntrial)
        )
    
 
    def _plottrial(self, index, noiseKey):
        ''' graphe de Nombre d'essais'''
        df_filtred = self.lv2_model.df_runs_filtred
        
        # Si pas de données de tâches pour les configurations sélectionnées
        if df_filtred.empty:
            return pn.pane.Markdown(f"Mutual Information : Pas de données complètes d'information mutuelle disponibles pour les configurations sélectionnées.")

        # Générer une palette de couleurs automatiquement selon le nombre de configurations
        fig = go.Figure()
        
        # Ajouter une trace pour chaque configuration et tâche
        for i, config in enumerate(index):
            # Filtrer les données pour chaque configuration
            matching_runs = df_filtred.index[df_filtred['Command_id'] == config]
            config_data = df_filtred.loc[matching_runs]
            
            # Vérifier que les colonnes existent et ne sont pas toutes NaN
            if noiseKey not in config_data.columns or "Mutual Information.n_trials" not in config_data.columns:
                continue  # Colonnes manquantes

            config_data = config_data.dropna(subset=[noiseKey, "Mutual Information.n_trials"])

            if config_data.empty:
                continue  # Plus de données après nettoyage
            
            snr = config_data[noiseKey]
            n_trials = config_data["Mutual Information.n_trials"]         
                
            fig.add_trace(go.Scatter(
                x=snr, y=n_trials,
                mode='markers',
                line=dict(width=2, color=next(self.colors)),
                name=f"Nombre d'essais - {config}"  
            ))
        
        if not fig.data:
            return pn.pane.Markdown(
                "Mutual Information : Données insuffisantes (valeurs NaN ou colonnes manquantes) pour les configurations sélectionnées."
            )
                
        # Configuration de la mise en page avec Range Slider et Range Selector
        fig.update_layout(
            title="Nombre d'essais en fonction du SNR pour chaque configuration",
            xaxis=dict(
                title=f"Niveau de Bruit (SNR) : {noiseKey}",
                rangeslider=dict(visible=True),
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1dB", step="all", stepmode="backward"),
                        dict(count=5, label="5dB", step="all", stepmode="backward"),
                        dict(count=10, label="10dB", step="all", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            ),
            yaxis=dict(
                title="Nombre d'essais",
            ),
            legend_title="Configurations",
            template="plotly_white",
            height=600,
            showlegend=True,
            margin=dict(t=70, b=50, l=50, r=10)
        )
        
        return  pn.pane.Plotly(fig, sizing_mode="stretch_width")

##################################### Niveau 3 : Commande ####################################

###################################################
## Gestion des données niveau 3 sélection unique ##
###################################################

class ConfigUniqueSelector(pn.viewable.Viewer):
    model = param.ClassSelector(class_=ConfigUniqueModel)

    def __init__(self, **params):
        super().__init__(**params)

        # RadioBoxGroup initialisé avec les alias disponibles
        self.selector = pn.widgets.RadioBoxGroup(
            name='Configurations',
            options=self.model.options_alias,
            value=self.model.alias() if self.model.alias() != '-' else None,
            inline=False
        )

        # Lorsque l'utilisateur change la sélection, on met à jour self.model.value
        self.selector.param.watch(self._sync_model_from_selector, 'value')

    def _sync_model_from_selector(self, event):
        """Binde la sélection (alias) vers le model.value."""
        if event.new:
            self.model.value_by_alias(event.new)
        else:
            self.model.value = None

    @param.depends('model.options', watch=True)
    def _sync_selector_from_model(self, event=None):
        alias = self.model.alias()
        opts = self.model.options_alias
        self.selector.options  = opts
        # Si l'alias du model n'est pas dans les options, on désactive
        if not alias == '-':
            self.selector.value = alias
            self.selector.disabled = False
        else:
            self.selector.value = None
            self.selector.disabled = True

    def __panel__(self):
        return pn.Column(
            pn.pane.Markdown("**Configurations :**"),
            self.selector
        )


####################################
## Affichage des journeaux d'exec ##
####################################

class LogViewer(pn.viewable.Viewer):
    unique_conf_model = param.ClassSelector(default=None, class_=ConfigUniqueModel)
    
    def __init__(self, **params):
        super().__init__(**params)
        
        self.output_pane = pn.pane.Markdown("Sélectionnez une configuration pour voir les fichiers.")
        self.radioBoutton = ConfigUniqueSelector(name="One Configuration Selection", model= self.unique_conf_model)
        
        self.date_selector = pn.widgets.Select(name="Date d'exécution", options=[], visible=False)
        self.date_selector.param.watch(self._update_log_on_date_change, "value")

    @param.depends('unique_conf_model.value', watch=True)
    def _update_dates(self, event=None):
        self.date_selector.options = self.unique_conf_model.options_dates
        self.date_selector.value = self.unique_conf_model.date
        self.output_pane.object = self.unique_conf_model.log

        if not self.unique_conf_model.date is None :
            self.date_selector.visible = True
        else:
            self.date_selector.visible = False

    def _update_log_on_date_change(self, event=None):
        self.unique_conf_model.date = self.date_selector.value
        self.output_pane.object = self.unique_conf_model.log

    def __panel__(self):
        # Affichage du sélecteur et des onglets
        return pn.Column(
            self.radioBoutton,
            self.date_selector,
            self.output_pane,
            sizing_mode="stretch_width")

######################
## Graphe de tâches ##
######################

class Tasks_Histogramme(pn.viewable.Viewer):
    # Paramètres configurables
    df = param.DataFrame(doc="Le dataframe contenant les données")
    multi_choice_widget = param.ClassSelector(default=None, class_=pn.viewable.Viewer, doc="Widget MultiChoice")
    noiseScale = param.ClassSelector(default=None, class_=pn.viewable.Viewer,doc="Choix de l'échelle de bruit par passage du label de la colonne")

    def __init__(self, **params):
        super().__init__(**params)
        
        self.button_time_perc = pn.widgets.Toggle(name='%', value=True)
        self.button_time_perc.param.watch(self.changeIcon, 'value')
        self.ListBouton = pn.Column(
            pn.widgets.TooltipIcon(value="Affichage des temps des tâches en milli-seconde ou en %."), 
            self.button_time_perc,
            width=50)
        self.graphPanel = pn.bind(self._plot_task_data, self.button_time_perc, self.multi_choice_widget.param.value, self.noiseScale.param.value)
        
    def changeIcon(self, event) :
        if event.new : 
            self.button_time_perc.name = '%'
        else :
            self.button_time_perc.name = '⏱'
    
    def __panel__(self):
        return pn.Row(self.ListBouton, self.graphPanel)
    
    def _plot_task_data(self,percent, index, noiseKey):
        db = pn.state.cache['db']
        
        
        if index is None :
            df_filtred = self.df
        else :
            df_filtred = self.df.loc[index] 
        
        if df_filtred.empty:
            self.button_time_perc.disabled=True
            return pn.pane.Markdown(f"Graphes de Tâches : Pas de données de tâches disponibles pour les configurations sélectionnées.")
        else : 
            self.button_time_perc.disabled=False
            
        if percent :
            y_label = ('Time', 'Durée')
        else :
            y_label = ('Perc','Durée (%)')
        
        
        
        # Pivot des données pour que chaque combinaison Config_Hash + Signal Noise Ratio(SNR).Eb/N0(dB) ait des colonnes pour les temps des tâches
        pivot_df = df_filtred.pivot_table(
            values=y_label[0], 
            index=['Config_Hash', noiseKey], 
            columns='Task',
            aggfunc='sum', 
            fill_value=0
        )

        # Générer une palette de couleurs automatiquement selon le nombre de configurations
        colors = px.colors.qualitative.Plotly[:len(index) * len(df_filtred['Task'].unique())]

        # Initialiser la figure Plotly
        fig = go.Figure()
        
        # Ajouter chaque tâche comme une barre empilée
        for task in pivot_df.columns:
            fig.add_trace(go.Bar(
                x=pivot_df.index.map(lambda x: f"{db['commands'].loc[x[0], 'Config_Alias']} - SNR: {x[1]}"),  # Combinaison Config_Hash + SNR comme étiquette
                y=pivot_df[task],
                name=task
            ))

        # Configuration de la mise en page
        fig.update_layout(
            barmode='stack',
            title=f"Temps des tâches par Configuration et Niveau de Bruit  : {noiseKey}",
            xaxis_title="Configuration et Niveau de Bruit",
            yaxis_title=y_label[1],
            xaxis=dict(tickangle=25),  # Rotation des étiquettes de l'axe x
            template="plotly_white",
            height=900,
            showlegend=True,
            margin=dict(t=70, b=50, l=50, r=10)
            
        )
        return pn.pane.Plotly(fig, sizing_mode="stretch_width")


###############################################################################################################################################################################################################################################################
#####################################                                                                      Input                                                                                                      ####################################
###############################################################################################################################################################################################################################################################

###########################################
## Chargement des données depuis Gitlab ###
###########################################

GITLAB_PACKAGE_URL = "https://gitlab.inria.fr/api/v4/projects/1420/packages/generic/gitlab-elk-export/latest/"

async def load_table(name: str, fmt: str = "parquet") -> pd.DataFrame:
    url = f"{GITLAB_PACKAGE_URL}{name}.{fmt}"

    try:
        if IS_PYODIDE :
            response = await pyfetch(url)
            data = await response.bytes()
        elif IS_PANEL_CONVERT :
            with open_url(url) as response:
                data = response.read()
        else:
            import httpx
            import pyarrow.parquet as pq

            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.content

        buf = BytesIO(data)

        if fmt == "parquet":
            table = pq.read_table(source=buf)
            return table.to_pandas()
        elif fmt == "json":
            try:
                return pd.read_json(buf, orient="records", lines=True)
            except ValueError:
                buf.seek(0)
                return pd.read_json(buf, orient="records")
        else:
            print(f"❌ Format non supporté : {fmt}")
            return pd.DataFrame()

    except Exception as e:
        print(f"❌ Erreur lors du chargement de {name}.{fmt} : {e}")
        return pd.DataFrame()

      
async def load_data():
    if IS_PYODIDE or IS_PANEL_CONVERT:
        fmt='json'
    else:    
        fmt = 'parquet'
    
    df_commands = await load_table('command', fmt=fmt)
    df_commands.set_index('Command_id', inplace=True)
    

    # df_meta = await load_table('meta', fmt=fmt)
    # df_meta.set_index('meta_id', inplace=True)


    df_param = await load_table('parameters', fmt=fmt)
    df_param.set_index('param_id', inplace=True)

    df_tasks = await load_table('tasks', fmt=fmt)
    df_tasks.set_index('RUN_id', inplace=True)

    df_runs = await load_table('runs', fmt=fmt)
    df_runs.set_index('RUN_id', inplace=True)

    df_git = await load_table('git', fmt=fmt)
    df_git.set_index('sha1', inplace=True)
    # df_git['date'] = pd.to_datetime(df_git['date'], utc=True)

    df_log = await load_table('logs', fmt=fmt)
    df_log.set_index('Log_id', inplace=True)

    # Générer Config_Alias depuis config
    df_commands['Config_Alias'] = df_commands.index + " : " + df_commands['Command_short'] + "_" + df_commands['sha1']
    config_aliases = dict(zip(df_commands.index, df_commands['Config_Alias']))

    pn.state.cache['db'] =  {
        "commands": df_commands,
        "tasks": df_tasks,
        "runs": df_runs,
        "git": df_git,
        # "meta": df_meta,
        "param": df_param,
        "config_aliases": config_aliases,
        "logs": df_log
    }

    # Afficher les types de données des DataFrames pour générer le typage des fichiers json depuis les parquets (utile pour le développement)
    if fmt == 'parquet':
        with open('output_typing_code.py.generate', 'w', encoding='utf-8') as f:
            for name, df in pn.state.cache['db'].items():
                if name != 'config_aliases':
                    f.write(generate_typing_code(df, name))
                    f.write('\n\n') 
    else :
        apply_typing_code()

def generate_typing_code(df, df_name="df"):
    ''' Génère du code Python exécutable pour forcer le typage des colonnes dans pn.state.cache["db"][df_name] '''
    lines = [
        f"# Typage pour {df_name}",
        f"{df_name} = pn.state.cache['db']['{df_name}']"
    ]
    
    for col in df.columns:
        dtype = df[col].dtype

        if pd.api.types.is_integer_dtype(dtype):
            lines.append(f"{df_name}['{col}'] = pd.to_numeric({df_name}['{col}'], errors='coerce').astype('Int64')")
        elif pd.api.types.is_float_dtype(dtype):
            lines.append(f"{df_name}['{col}'] = pd.to_numeric({df_name}['{col}'], errors='coerce')")
        elif pd.api.types.is_bool_dtype(dtype):
            lines.append(f"{df_name}['{col}'] = {df_name}['{col}'].astype(bool)")
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            lines.append(f"{df_name}['{col}'] = pd.to_datetime({df_name}['{col}'], errors='coerce')")
        else:
            lines.append(f"{df_name}['{col}'] = {df_name}['{col}'].astype(str)")

    # Réassigner dans le cache (non strictement nécessaire mais plus explicite)
    lines.append(f"pn.state.cache['db']['{df_name}'] = {df_name}")

    return "\n".join(lines)

    
def apply_typing_code():
    ''' Applique le typage des données  (copier coller du résultat de generate_typing_code) ''' 
    # Typage pour commands
    commands = pn.state.cache['db']['commands']
    commands['Command'] = commands['Command'].astype(str)
    commands['sha1'] = commands['sha1'].astype(str)
    commands['Command_short'] = commands['Command_short'].astype(str)
    commands['param_id'] = commands['param_id'].astype(str)
    commands['Config_Alias'] = commands['Config_Alias'].astype(str)
    pn.state.cache['db']['commands'] = commands

    # Typage pour tasks
    tasks = pn.state.cache['db']['tasks']
    pn.state.cache['db']['tasks'] = tasks

    # Typage pour runs
    runs = pn.state.cache['db']['runs']
    runs['log_hash'] = runs['log_hash'].astype(str)
    runs['Date_Execution'] = pd.to_datetime(runs['Date_Execution'], errors='coerce')
    runs['Bit Error Rate (BER) and Frame Error Rate (FER).BE'] = pd.to_numeric(runs['Bit Error Rate (BER) and Frame Error Rate (FER).BE'], errors='coerce').astype('Int64')
    runs['Bit Error Rate (BER) and Frame Error Rate (FER).BER'] = pd.to_numeric(runs['Bit Error Rate (BER) and Frame Error Rate (FER).BER'], errors='coerce')
    runs['Bit Error Rate (BER) and Frame Error Rate (FER).FE'] = pd.to_numeric(runs['Bit Error Rate (BER) and Frame Error Rate (FER).FE'], errors='coerce').astype('Int64')
    runs['Bit Error Rate (BER) and Frame Error Rate (FER).FER'] = pd.to_numeric(runs['Bit Error Rate (BER) and Frame Error Rate (FER).FER'], errors='coerce')
    runs['Bit Error Rate (BER) and Frame Error Rate (FER).FRA'] = pd.to_numeric(runs['Bit Error Rate (BER) and Frame Error Rate (FER).FRA'], errors='coerce').astype('Int64')
    runs['Global throughputand elapsed time.SIM_THR(Mb/s)'] = pd.to_numeric(runs['Global throughputand elapsed time.SIM_THR(Mb/s)'], errors='coerce')
    runs['Global throughputand elapsed time.elapse_time(ns)'] = pd.to_numeric(runs['Global throughputand elapsed time.elapse_time(ns)'], errors='coerce')
    runs['Signal Noise Ratio(SNR).Eb/N0(dB)'] = pd.to_numeric(runs['Signal Noise Ratio(SNR).Eb/N0(dB)'], errors='coerce')
    runs['Signal Noise Ratio(SNR).Es/N0(dB)'] = pd.to_numeric(runs['Signal Noise Ratio(SNR).Es/N0(dB)'], errors='coerce')
    runs['Signal Noise Ratio(SNR).Sigma'] = pd.to_numeric(runs['Signal Noise Ratio(SNR).Sigma'], errors='coerce')
    runs['source.type'] = runs['source.type'].astype(str)
    runs['id'] = runs['id'].astype(str)
    runs['url'] = runs['url'].astype(str)
    runs['status'] = runs['status'].astype(str)
    runs['job_id'] = runs['job_id'].astype(str)
    runs['job_name'] = runs['job_name'].astype(str)
    runs['Signal Noise Ratio(SNR).Event Probability'] = pd.to_numeric(runs['Signal Noise Ratio(SNR).Event Probability'], errors='coerce')
    runs['Mutual Information.MI'] = pd.to_numeric(runs['Mutual Information.MI'], errors='coerce')
    runs['Mutual Information.MI_max'] = pd.to_numeric(runs['Mutual Information.MI_max'], errors='coerce')
    runs['Mutual Information.MI_min'] = pd.to_numeric(runs['Mutual Information.MI_min'], errors='coerce')
    runs['Mutual Information.n_trials'] = pd.to_numeric(runs['Mutual Information.n_trials'], errors='coerce')
    runs['Signal Noise Ratio(SNR).Received Optical'] = pd.to_numeric(runs['Signal Noise Ratio(SNR).Received Optical'], errors='coerce')
    runs['Command_id'] = runs['Command_id'].astype(str)
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

    # Typage pour param
    param = pn.state.cache['db']['param']
    param['Channel.Add users'] = param['Channel.Add users'].astype(str)
    param['Channel.Complex'] = param['Channel.Complex'].astype(str)
    param['Channel.Implementation'] = param['Channel.Implementation'].astype(str)
    param['Channel.Type'] = param['Channel.Type'].astype(str)
    param['Codec.Code rate'] = param['Codec.Code rate'].astype(str)
    param['Codec.Codeword size (N_cw)'] = param['Codec.Codeword size (N_cw)'].astype(str)
    param['Codec.Frame size (N)'] = param['Codec.Frame size (N)'].astype(str)
    param['Codec.Info. bits (K)'] = param['Codec.Info. bits (K)'].astype(str)
    param['Codec.Type'] = param['Codec.Type'].astype(str)
    param['Decoder.Correction power (T)'] = param['Decoder.Correction power (T)'].astype(str)
    param['Decoder.Galois field order (m)'] = param['Decoder.Galois field order (m)'].astype(str)
    param['Decoder.Implementation'] = param['Decoder.Implementation'].astype(str)
    param['Decoder.Systematic'] = param['Decoder.Systematic'].astype(str)
    param['Decoder.Type (D)'] = param['Decoder.Type (D)'].astype(str)
    param['Encoder.Systematic'] = param['Encoder.Systematic'].astype(str)
    param['Encoder.Type'] = param['Encoder.Type'].astype(str)
    param['Modem.Bits per symbol'] = param['Modem.Bits per symbol'].astype(str)
    param['Modem.Implementation'] = param['Modem.Implementation'].astype(str)
    param['Modem.Sigma square'] = param['Modem.Sigma square'].astype(str)
    param['Modem.Type'] = param['Modem.Type'].astype(str)
    param['Monitor.Compute mutual info'] = param['Monitor.Compute mutual info'].astype(str)
    param['Monitor.Frame error count (e)'] = param['Monitor.Frame error count (e)'].astype(str)
    param['Monitor.Lazy reduction'] = param['Monitor.Lazy reduction'].astype(str)
    param['Simulation.Bad frames replay'] = param['Simulation.Bad frames replay'].astype(str)
    param['Simulation.Bad frames tracking'] = param['Simulation.Bad frames tracking'].astype(str)
    param['Simulation.Bit rate'] = param['Simulation.Bit rate'].astype(str)
    param['Simulation.Code type (C)'] = param['Simulation.Code type (C)'].astype(str)
    param['Simulation.Coded monitoring'] = param['Simulation.Coded monitoring'].astype(str)
    param['Simulation.Coset approach (c)'] = param['Simulation.Coset approach (c)'].astype(str)
    param['Simulation.Date (UTC)'] = param['Simulation.Date (UTC)'].astype(str)
    param['Simulation.Debug mode'] = param['Simulation.Debug mode'].astype(str)
    param['Simulation.Git version'] = param['Simulation.Git version'].astype(str)
    param['Simulation.Inter frame level'] = param['Simulation.Inter frame level'].astype(str)
    param['Simulation.Json export'] = param['Simulation.Json export'].astype(str)
    param['Simulation.Multi-threading (t)'] = param['Simulation.Multi-threading (t)'].astype(str)
    param['Simulation.Noise range'] = param['Simulation.Noise range'].astype(str)
    param['Simulation.Noise type (E)'] = param['Simulation.Noise type (E)'].astype(str)
    param['Simulation.Seed'] = param['Simulation.Seed'].astype(str)
    param['Simulation.Statistics'] = param['Simulation.Statistics'].astype(str)
    param['Simulation.Type'] = param['Simulation.Type'].astype(str)
    param['Simulation.Type of bits'] = param['Simulation.Type of bits'].astype(str)
    param['Simulation.Type of reals'] = param['Simulation.Type of reals'].astype(str)
    param['Source.Implementation'] = param['Source.Implementation'].astype(str)
    param['Source.Info. bits (K_info)'] = param['Source.Info. bits (K_info)'].astype(str)
    param['Source.Type'] = param['Source.Type'].astype(str)
    param['Terminal.Enabled'] = param['Terminal.Enabled'].astype(str)
    param['Terminal.Frequency (ms)'] = param['Terminal.Frequency (ms)'].astype(str)
    param['Terminal.Show Sigma'] = param['Terminal.Show Sigma'].astype(str)
    param['Quantizer.Fixed-point config.'] = param['Quantizer.Fixed-point config.'].astype(str)
    param['Quantizer.Implementation'] = param['Quantizer.Implementation'].astype(str)
    param['Quantizer.Type'] = param['Quantizer.Type'].astype(str)
    param['Simulation.Type of quant. reals'] = param['Simulation.Type of quant. reals'].astype(str)
    param['Decoder.H matrix path'] = param['Decoder.H matrix path'].astype(str)
    param['Decoder.H matrix reordering'] = param['Decoder.H matrix reordering'].astype(str)
    param['Decoder.Num. of iterations (i)'] = param['Decoder.Num. of iterations (i)'].astype(str)
    param['Decoder.Stop criterion (syndrome)'] = param['Decoder.Stop criterion (syndrome)'].astype(str)
    param['Decoder.Stop criterion depth'] = param['Decoder.Stop criterion depth'].astype(str)
    param['Decoder.Weighting factor'] = param['Decoder.Weighting factor'].astype(str)
    param['Encoder.G build method'] = param['Encoder.G build method'].astype(str)
    param['Encoder.H matrix path'] = param['Encoder.H matrix path'].astype(str)
    param['Encoder.H matrix reordering'] = param['Encoder.H matrix reordering'].astype(str)
    param['Decoder.Bernouilli probas'] = param['Decoder.Bernouilli probas'].astype(str)
    param['CRC.Implementation'] = param['CRC.Implementation'].astype(str)
    param['CRC.Polynomial (hexadecimal)'] = param['CRC.Polynomial (hexadecimal)'].astype(str)
    param['CRC.Size (in bit)'] = param['CRC.Size (in bit)'].astype(str)
    param['CRC.Type'] = param['CRC.Type'].astype(str)
    param['Decoder.Adaptative mode'] = param['Decoder.Adaptative mode'].astype(str)
    param['Decoder.Max num. of lists (L)'] = param['Decoder.Max num. of lists (L)'].astype(str)
    param['Decoder.Polar node types'] = param['Decoder.Polar node types'].astype(str)
    param['Decoder.SIMD strategy'] = param['Decoder.SIMD strategy'].astype(str)
    param['Frozen bits generator.Noise'] = param['Frozen bits generator.Noise'].astype(str)
    param['Frozen bits generator.Type'] = param['Frozen bits generator.Type'].astype(str)
    param['Puncturer.Type'] = param['Puncturer.Type'].astype(str)
    param['Decoder.Node type'] = param['Decoder.Node type'].astype(str)
    param['Frozen bits generator MK.Noise'] = param['Frozen bits generator MK.Noise'].astype(str)
    param['Frozen bits generator MK.Type'] = param['Frozen bits generator MK.Type'].astype(str)
    param['Polar code.Kernel'] = param['Polar code.Kernel'].astype(str)
    param['Decoder.Min type'] = param['Decoder.Min type'].astype(str)
    param['Interleaver.Seed'] = param['Interleaver.Seed'].astype(str)
    param['Interleaver.Type'] = param['Interleaver.Type'].astype(str)
    param['Interleaver.Uniform'] = param['Interleaver.Uniform'].astype(str)
    param['Encoder.Buffered'] = param['Encoder.Buffered'].astype(str)
    param['Polar code.Kernels'] = param['Polar code.Kernels'].astype(str)
    param['Polar code.Stages'] = param['Polar code.Stages'].astype(str)
    param['Puncturer.Pattern'] = param['Puncturer.Pattern'].astype(str)
    param['Codec.Symbols Codeword size'] = param['Codec.Symbols Codeword size'].astype(str)
    param['Codec.Symbols Source size'] = param['Codec.Symbols Source size'].astype(str)
    param['Decoder.Max type'] = param['Decoder.Max type'].astype(str)
    param['Decoder.Polynomials'] = param['Decoder.Polynomials'].astype(str)
    param['Decoder.Standard'] = param['Decoder.Standard'].astype(str)
    param['Encoder.Polynomials'] = param['Encoder.Polynomials'].astype(str)
    param['Encoder.Standard'] = param['Encoder.Standard'].astype(str)
    param['Encoder.Tail length'] = param['Encoder.Tail length'].astype(str)
    param['Decoder.Num. of lists (L)'] = param['Decoder.Num. of lists (L)'].astype(str)
    param['Decoder.Normalize factor'] = param['Decoder.Normalize factor'].astype(str)
    param['Source.Auto reset'] = param['Source.Auto reset'].astype(str)
    param['Source.Fifo mode'] = param['Source.Fifo mode'].astype(str)
    param['Source.Path'] = param['Source.Path'].astype(str)
    param['Flip and check.Enabled'] = param['Flip and check.Enabled'].astype(str)
    param['Scaling factor.Enabled'] = param['Scaling factor.Enabled'].astype(str)
    param['Scaling factor.SF iterations'] = param['Scaling factor.SF iterations'].astype(str)
    param['Scaling factor.Scaling factor (SF)'] = param['Scaling factor.Scaling factor (SF)'].astype(str)
    param['Flip and check.FNC ite max'] = param['Flip and check.FNC ite max'].astype(str)
    param['Flip and check.FNC ite min'] = param['Flip and check.FNC ite min'].astype(str)
    param['Flip and check.FNC ite step'] = param['Flip and check.FNC ite step'].astype(str)
    param['Flip and check.FNC q'] = param['Flip and check.FNC q'].astype(str)
    param['Modem.Max type'] = param['Modem.Max type'].astype(str)
    param['Frozen bits generator.Path'] = param['Frozen bits generator.Path'].astype(str)
    param['Modem.Codebook'] = param['Modem.Codebook'].astype(str)
    param['Modem.Number of iterations'] = param['Modem.Number of iterations'].astype(str)
    param['Modem.Psi function'] = param['Modem.Psi function'].astype(str)
    param['Interleaver.Number of columns'] = param['Interleaver.Number of columns'].astype(str)
    param['Channel.Block fading policy'] = param['Channel.Block fading policy'].astype(str)
    param['Modem.CPM L memory'] = param['Modem.CPM L memory'].astype(str)
    param['Modem.CPM h index'] = param['Modem.CPM h index'].astype(str)
    param['Modem.CPM mapping'] = param['Modem.CPM mapping'].astype(str)
    param['Modem.CPM sampling factor'] = param['Modem.CPM sampling factor'].astype(str)
    param['Modem.CPM standard'] = param['Modem.CPM standard'].astype(str)
    param['Modem.CPM wave shape'] = param['Modem.CPM wave shape'].astype(str)
    param['Decoder.Num. of flips'] = param['Decoder.Num. of flips'].astype(str)
    param['Interleaver.Path'] = param['Interleaver.Path'].astype(str)
    param['Simulation.Global iterations (I)'] = param['Simulation.Global iterations (I)'].astype(str)
    param['Modem.ROP estimation'] = param['Modem.ROP estimation'].astype(str)
    param['Simulation.PDF path'] = param['Simulation.PDF path'].astype(str)
    pn.state.cache['db']['param'] = param

    # Typage pour logs
    logs = pn.state.cache['db']['logs']
    logs['log'] = logs['log'].astype(str)
    logs['hash'] = logs['hash'].astype(str)
    logs['filename'] = logs['filename'].astype(str)
    logs['Date_Execution'] = logs['Date_Execution'].astype(str)
    pn.state.cache['db']['logs'] = logs






# Performance par niveau de bruit pour les configurations sélectionnées
def plot_performance_metrics_plotly(configs, noiseScale):
    # Si aucune configuration n'est sélectionnée
    if not configs:
        return pn.pane.Markdown("Veuillez sélectionner au moins une configuration pour afficher les performances.")
    db = pn.state.cache['db']
    filtered_df_runs = db['runs'][db['runs']["Command_id"].isin(configs)]
    if filtered_df_runs.empty:
        return pn.pane.Markdown("Pas de données de performance disponibles pour les configurations sélectionnées.")
    
    filtered_df_runs = filtered_df_runs.sort_values(by=noiseScale, ascending=True)
    
    
    fig = go.Figure()
    
    # Générer une palette de couleurs automatiquement selon le nombre de configurations
    colors = px.colors.qualitative.Plotly[:len(configs)]  # Choisir des couleurs depuis Plotly, ajustées à la taille de configs
    
    for i, config in enumerate(configs):
        config_data = filtered_df_runs.loc[filtered_df_runs["Command_id"] == config]
        snr = config_data[noiseScale]
        ber = config_data['Bit Error Rate (BER) and Frame Error Rate (FER).BER']
        fer = config_data['Bit Error Rate (BER) and Frame Error Rate (FER).FER']
        
        # Trace BER (ligne pleine avec marqueurs)
        fig.add_trace(go.Scatter(
            x=snr, y=ber, mode='lines+markers', name=f"BER - {config}",
            line=dict(width=2, color=colors[i]),
            marker=dict(symbol='circle', size=6)
        ))
        
        # Trace FER (ligne pointillée avec marqueurs)
        fig.add_trace(go.Scatter(
            x=snr, y=fer, mode='lines+markers', name=f"FER - {config}",
            line=dict(width=2, dash='dash', color=colors[i]),
            marker=dict(symbol='x', size=6)
        ))
    
    
    # Configuration de la mise en page avec Range Slider et Range Selector
    fig.update_layout(
        title="BER et FER en fonction du SNR pour chaque configuration",
        xaxis=dict(
            title=f"Niveau de Bruit (SNR) : {noiseScale}",
            rangeslider=dict(visible=True),  # Activation du Range Slider
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1dB", step="all", stepmode="backward"),
                    dict(count=5, label="5dB", step="all", stepmode="backward"),
                    dict(count=10, label="10dB", step="all", stepmode="backward"),
                    dict(step="all")
                ])
            )
        ),
        yaxis=dict(
            title="Taux d'Erreur",
            type='log'
        ),
        legend_title="Configurations",
        template="plotly_white",
        height=600,
        showlegend=True,
        margin=dict(t=70, b=50, l=50, r=10)
    )
    
    return pn.pane.Plotly(fig, sizing_mode="stretch_width")





# Configurer argparse pour gérer les arguments en ligne de commande
def parse_arguments():
    parser = argparse.ArgumentParser(description="Tableau de bord des commits.")
    parser.add_argument('-l', '--local', action="store_true", help="Local affiche le tableau de bord dans le navigateur, son absence permet son export.")  # on/off flag
    return parser.parse_args()

 # Utiliser des valeurs par défaut dans le cas d'un export qui ne supporte pas argparse
class DefaultArgs:
    local = False
args = DefaultArgs()
if __name__ == "__main__":
    args = parse_arguments()  # Appel unique de argparse ici

#######################
## Paramêtre du site ##
#######################


noiseScale = NoiseScale(noise_label= noise_label)

###############################################################################################################################################################################################################################################################
#####################################                                                                      Assemblage                                                                                                      ####################################
###############################################################################################################################################################################################################################################################



############################
## Assemblage du panel git ##
############################

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
        self.indicators = GitIndicators(df_git=db['git'], df_commands=db['commands'], filter_model=self.git_filter)
        self.perfgraph = PerformanceByCommit(git_filter=self.git_filter, command_filter=self.command_filter)
        self.research_config_filter = Research_config_filter(command_filter=self.command_filter)

    def __panel__(self):
        return pn.Column(
            self.indicators,
            self.date_slider,
            self.table,
            self.code_selector,
            self.research_config_filter,
            self.perfgraph,
            sizing_mode="stretch_width"
        )

    def update_command_table(self, event=None):
        self.command_table.value = self.command_filter.get_filtered_df()



def init_dashboard():
    db = pn.state.cache['db']

    git_filter = GitFilterModel(df_git=db['git'])

    #ajout du code aux commandes
    merged_df = db['commands'].merge(db['param'], #[['Simulation.Code type (C)']], 
                                    left_on='param_id',
                                    right_index=True,
                                    how='left')
    merged_df.rename(columns={'Simulation.Code type (C)': 'code'}, inplace=True)

    command_filter = CommandFilterModel(df_commands=merged_df, git_filter=git_filter)

    panelCommit = PanelCommit(command_filter=command_filter, git_filter=git_filter)

    ##################################### Config ####################################

    lvl2_filter = Lvl2_Filter_Model(command_filter=command_filter)
    config_panel = ConfigPanel(lv2_model=lvl2_filter)

    mi_panel = pn.Column(
        Mutual_information_Panels(
            lv2_model = lvl2_filter,
            noiseScale =noiseScale
        ),
        scroll=True, height=700
    )

    # panel des configs
    panelConfig = pn.Row(
        pn.Column(
            config_panel,
            TableConfig(lv2_filter=lvl2_filter, meta=False),
            pn.Tabs(
                ('ılıılıılıılıılıılı BER/FER', pn.bind(plot_performance_metrics_plotly, lvl2_filter.param.value, noiseScale.param.value)),
                ('⫘⫘⫘ Mutual information', mi_panel)
            ),
            sizing_mode="stretch_width"
        )
    )

    ##################################### Performance par niveau de SNR ####################################

    unique_model = ConfigUniqueModel(lv2_model=lvl2_filter)

    # # Histogramme des temps des jobs
    # task_Time_Histogramme = Tasks_Histogramme(
    #     multi_choice_widget = config_selector,
    #     df = db['tasks'],
    #     noiseScale = noiseScale
    # ) 

    # plot_debit = Panel_graph_envelope(
    #     multi_choice_widget = config_selector,
    #     df = db['tasks'],
    #     lab ="Measured throughput Average", 
    #     labmin="Measured throughput Mininmum", 
    #     labmax="Measured throughputMaximum", 
    #     lab_group='Task',
    #     Ytitle = "Débit",
    #     noiseScale = noiseScale
    # )

    # plot_latence = Panel_graph_envelope(
    #     multi_choice_widget = config_selector,
    #     df = db['tasks'],
    #     lab ="Measured latency Average", 
    #     labmin="Measured latency Mininmum", 
    #     labmax="Measured latency Maximum", 
    #     lab_group='Task',
    #     Ytitle = "Latence",
    #     noiseScale = noiseScale
    #)    

    panel_par_config = pn.Column(
        pn.pane.HTML("<div style='font-size: 20px;background-color: #e0e0e0; padding: 5px;line-height : 0px;'><h2> ✏️ Logs</h2></div>"),
        LogViewer(unique_conf_model=unique_model),
        #TableConfig(lv2_filter=lvl2_filter, meta=True),
        # task_Time_Histogramme,
        # plot_latence,
        # plot_debit,
        sizing_mode="stretch_width"
    )

    ##################################### Panel Données ####################################

    # Widgets d'affichage des informations
    config_count = pn.indicators.Number(name="Configurations en base", value=db['commands'].shape[0] if not db['commands'].empty else 0)

    #panel de la partie data
    panelData = pn.Column(config_count,
                        sizing_mode="stretch_width")

    # Layout du tableau de bord avec tout dans une colonne et des arrières-plans différents

    dashboard = pn.Column(
        
        pn.pane.HTML("<div style='font-size: 28px;background-color: #e0e0e0; padding: 10px;line-height : 0px;'><h2> ✏️ Niveau 1 : Evolution par commit </h2></div>"),
        panelCommit,
        pn.pane.HTML("<div style='font-size: 28px;background-color: #e0e0e0; padding: 10px;line-height : 0px;'><h2> ☎️ Niveau 2 : BER / FER </h2></div>"),
        panelConfig,
        pn.pane.HTML("<div style='font-size: 28px;background-color: #e0e0e0; padding: 10px;line-height : 0px;'><h2> ⚙️ Niveau 3 : Analyse à la commande</h2></div>"),
        panel_par_config,
        )

    paramSite = noiseScale
    
    dashboard= pn.template.FastListTemplate(
        title="Commits Dashboard",
        sidebar=[logo, paramSite,  pn.layout.Divider(), panelData],
        main=[dashboard],
        main_layout=None,
        accent=ACCENT,
        theme_toggle=False,
    )
    
    return dashboard
    # Lancer le tableau de bord


################################
## Démarage selon le contexte ##
################################

# Variables globales
dashboard = None

# Détection de l'environnement AVANT les définitions de fonctions
IS_PYODIDE = sys.platform == "emscripten"
IS_PANEL_CONVERT = os.getenv("PANEL_CONVERT") == "1"

async def startup():
    """Version asynchrone pour Pyodide"""
    global dashboard
    await load_data()
    dashboard = init_dashboard()
    dashboard.servable()
    
    if IS_PYODIDE: 
        await pn.io.pyodide.write_doc()
    
    
    return dashboard

def convert_startup():
    """Version synchrone pour Panel Convert"""
    global dashboard
    
    # Créer une nouvelle boucle d'événements si nécessaire
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Exécuter le chargement des données
    loop.run_until_complete(load_data())
    dashboard = init_dashboard()
    dashboard.servable()
    
    return dashboard

def launch():
    """Point d'entrée principal"""
    global dashboard
    
    if IS_PYODIDE:
        # Pour Pyodide, utiliser onload
        print("Mode Pyodide détecté")
        pn.state.onload(startup)
        
    elif IS_PANEL_CONVERT:
        # Pour panel convert
        print("Mode Panel Convert détecté")
        convert_startup()
        
    else:
        # Mode développement local
        print("Mode développement local")
        asyncio.run(startup())
        pn.serve(dashboard, show=True, port=35489)

# Point d'entrée
print(ud.unidata_version)
launch()