from bokeh.settings import settings
settings.resources = 'inline'

import pandas as pd
import panel as pn
from datetime import datetime
import argparse
import re
try:
    from pyodide.http import open_url
except ModuleNotFoundError:
    # Utiliser un fallback avec urllib en environnement standard
    from urllib.request import urlopen as open_url
#import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import plotly.graph_objects as go
import param
from panel.viewable import Viewer
import unicodedata as ud
import itertools
from io import BytesIO

## lib non compatible avec pyodide (usage uniquement en local)
import httpx
import pyarrow.parquet as pq


print(ud.unidata_version)


##################################### Niveau Global ####################################

# Initialiser Panel
pn.extension("plotly", sizing_mode="stretch_width")  # Adapter la taille des widgets et graphiques à la largeur de l'écran


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




##################################### Niveau 1 : Git et perf global ####################################

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
    filtered = param.Parameter()

    def __init__(self, **params):
        super().__init__(**params)
        # Initialisation de 'code' avec toutes les valeurs possibles dans df_commands['code']
        all_codes = sorted(self.df_commands['code'].dropna().unique().tolist())
        self.param['code'].objects = all_codes
        self.param['code'].default = all_codes 
        self.codes = all_codes

    @param.depends('code', 'git_filter.filtered', watch=True)
    def _trigger(self):
        self.param.trigger('filtered')  


    def get_filtered_df(self):
        sha1_valids = self.git_filter.get_sha1_valids()
        df_filtered = self.df_commands[self.df_commands['sha1'].isin(sha1_valids)]
        if 'All' not in self.code:
            df_filtered = df_filtered[df_filtered['code'].isin(self.code)]
        return df_filtered

#################################
## Component pour le Panel Git ##
#################################
class DateRangeFilter(pn.viewable.Viewer):
    git_filter = param.ClassSelector(class_=GitFilterModel)

    def __init__(self, **params):
        super().__init__(**params)
        # Bornes extraites du DataFrame Git
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

################################################
## Gestion des données niveau 2 avec filtrage ##
################################################

class ConfigPanel(pn.viewable.Viewer):
    command_filter = param.ClassSelector(class_=CommandFilterModel)
    df = param.DataFrame()  # Contiendra les données filtrées en cache
    value = param.List(default=[], doc="Liste des index sélectionnés")

    def __init__(self, **params):
        if 'df' in params:
            raise ValueError("Le paramètre 'df' ne doit pas être initialisé dans le constructeur.")
        
        super().__init__(**params)

        self.config_selector = pn.widgets.MultiChoice(name="Sélectionnez les configurations", options=[])

        self.select_all_button = pn.widgets.Button(name="Tout sélectionner", button_type="success")
        self.clear_button = pn.widgets.Button(name="Tout désélectionner", button_type="warning")

        self.select_all_button.on_click(self.select_all_configs)
        self.clear_button.on_click(self.clear_configs)

        self.config_selector.param.watch(self._update_value, 'value')
        self.command_filter.param.watch(self._update_df, 'filtered')

        df = db['commands'].join(db['param'], on='param_id')
        
        self._update_df()

    def __panel__(self):
        return pn.Column(
            self.select_all_button,
            self.clear_button,
            self.config_selector,
            Research_config_filter(config_selector=self.config_selector, df=self.df),
        )

    def _update_df(self, *events):
        self.df = self.command_filter.get_filtered_df()
        options = self.df['Config_Alias'].dropna().unique().tolist()
        self.config_selector.options = options
        self.config_selector.value = []  # reset la sélection au besoin

    def _update_value(self, event=None):
        selected_configs = self.config_selector.value
        if selected_configs:
            self.value = self.df[self.df["Config_Alias"].isin(selected_configs)].index.tolist()
        else:
            self.value = []

    def select_all_configs(self, event=None):
        self.config_selector.value = self.config_selector.options

    def clear_configs(self, event=None):
        self.config_selector.value = []
            
# affichage de la sélection     
class TableConfig(pn.viewable.Viewer):
    df = param.DataFrame(doc="Le dataframe contenant les données")
    config_selector = param.ClassSelector(default=None, class_=pn.viewable.Viewer, doc="Widget MultiChoice")
    meta = param.Boolean(doc="affiche les Meta-données si Vrai, les paramètres de simmulation si faux")
    def __init__(self, **params):
        super().__init__(**params)
        self.tab =  pn.pane.DataFrame(self._prepare(), name='table.selected_config', index=False)

    def __panel__(self):
        return pn.Accordion( ("📥 Selected Configuration", self.tab))
    
    @param.depends('config_selector.value', watch=True)
    def table_selected_config_filter(self):
        self.tab.object = self._prepare()

    def _prepare(self):
        c_filter = self.df.loc[self.config_selector.value]
        
        if self.meta :
            filtered_df = db['meta'][db['meta'].index.isin(c_filter['meta_id'])]
        else :
            filtered_df = db['param'][db['param'].index.isin(c_filter['param_id'])]
        
        return filtered_df




class Panel_graph_envelope(pn.viewable.Viewer):
    # Paramètres configurables
    df = param.DataFrame(doc="Le dataframe contenant les données")
    lab = param.String(default="y", doc="Nom de la colonne pour l'axe Y")
    lab_group = param.String(default=None, doc="Nom de la colonne pour regrouper les données")
    labmin = param.String(default=None, doc="Nom de la colonne pour la valeur minimale")
    labmax = param.String(default=None, doc="Nom de la colonne pour la valeur maximale")
    Ytitle = param.String(default="Valeur", doc="Titre de l'axe Y")
    multi_choice_widget = param.ClassSelector(default=None, class_=pn.viewable.Viewer, doc="Panel de sélection des configurations")
    noiseScale = param.ClassSelector(default=None, class_=pn.viewable.Viewer,doc="Choix de l'échelle de bruit par passage du label de la colonne")

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
        self.graphPanel = pn.bind(self._plot_enveloppe_incertitude,self.button_env, self.multi_choice_widget.param.value, self.noiseScale.param.value)
        

    def __panel__(self):
        return pn.Row(self.ListBouton, self.graphPanel)
        
    def _plot_enveloppe_incertitude(self, show_envelope, index, noiseKey):    
     
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
    df = param.DataFrame(doc="Le dataframe contenant les données")
    index_selecter = param.ClassSelector(default=None, class_=pn.viewable.Viewer, doc="Widget MultiChoice")
    noiseScale = param.ClassSelector(default=None, class_=pn.viewable.Viewer,doc="Choix de l'échelle de bruit par passage du label de la colonne")

    def __init__(self, **params):
        super().__init__(**params)
        
        self.colors = itertools.cycle(px.colors.qualitative.Plotly)
        
        cols = ["Mutual Information.MI", "Mutual Information.MI_min", "Mutual Information.MI_max", "Mutual Information.n_trials"]
        self.df = self.df [ self.df[cols].notnull().any(axis=1) ]
        
        self.plot_mutual_information = Panel_graph_envelope(
            multi_choice_widget = self.index_selecter,
            df = self.df,
            lab   ="Mutual Information.MI", 
            labmin="Mutual Information.MI_min", 
            labmax="Mutual Information.MI_max", 
            Ytitle = "Information mutuelle",
            noiseScale = self.noiseScale
        )
        
        self.ListBouton = pn.Column(
            pn.widgets.TooltipIcon(value="Seuls les configuration avec des valeurs pour \"Mutual Information.MI\", \"Mutual Information.MI_min\", \"Mutual Information.MI_max\", \"Mutual Information.n_trials\" sont affichées. "), 
            width=50)
        self.mutual_information_ntrial = pn.bind(self._plottrial, self.index_selecter.param.value, self.noiseScale.param.value)

    def __panel__(self):
        return pn.Column(
            pn.widgets.TooltipIcon(value="Seuls les configuration avec des valeurs pour \"Mutual Information.MI\", \"Mutual Information.MI_min\", \"Mutual Information.MI_max\" sont affichées. "), 
            pn.Row(self.plot_mutual_information),
            pn.Row(self.ListBouton, self.mutual_information_ntrial)
        )
    
 
    def _plottrial(self, index, noiseKey): 
        if index is None :
            df_filtred = self.df
        else :
            df_filtred = self.df[self.df["Command_id"].isin(index)] 
        
        # Si pas de données de tâches pour les configurations sélectionnées
        if df_filtred.empty:
            return pn.pane.Markdown(f"Mutual Information : Pas de données complètes d'information mutuelle disponibles pour les configurations sélectionnées.")

        # Générer une palette de couleurs automatiquement selon le nombre de configurations
        fig = go.Figure()
        
        # Ajouter une trace pour chaque configuration et tâche
        for i, config in enumerate(index):
            # Filtrer les données pour chaque configuration
            config_data = df_filtred['Command_id'][config]
            snr = config_data[noiseKey]
            n_trials = config_data["Mutual Information.n_trials"]         
                
            fig.add_trace(go.Scatter(
                x=snr, y=n_trials,
                mode='markers',
                line=dict(width=2, color=next(self.colors)),
                name=f"Nombre d'essais - {config}"  
            ))
        
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

class Research_config_filter(pn.viewable.Viewer):
    # Paramètres configurables
    df = param.DataFrame(doc="Le dataframe contenant les données")
    config_selector = param.ClassSelector(default=None, class_=pn.widgets.MultiChoice, doc="Widget MultiChoice", allow_refs=False)
    def __init__(self, **params):
        super().__init__(**params)
        
        # Configuration initiale Tout
        self._config_allowed = {}
        for col in self.df.columns:
            self._config_allowed[col] = self.df[col].unique().tolist()
        
        # Filtrer les colonnes qui suivent le format "FAMILLE.nom"
        df_commands = self.df
        # Les colonnes suivantes ne doivent pas avoir de filtre
        columns_to_exclude = ['Config_Alias', 'param_id', 'meta_id', 'git_id', 'Command']

        df_commands = df_commands.drop(columns=columns_to_exclude, errors='ignore')
        
        family_columns = {}
        for col in df_commands.columns:
            match = re.match(r"(\w+)\.(\w+)", col)
            if match:
                family, name = match.groups()
                if family not in family_columns:
                    family_columns[family] = []
                family_columns[family].append(col)
            else:
                # Ajoute les colonnes sans famille dans une clé générale
                family_columns.setdefault("Autres", []).append(col)
        
        # Créer les widgets de sélection pour chaque famille
        family_widgets = {}
        for family, columns in family_columns.items():
            widgets = []
            for col in columns :
                options = df_commands[col].unique().tolist()
                is_disabled = len(options) == 1
                widget = pn.widgets.MultiChoice(name=col, options=options, value=options, disabled=is_disabled, css_classes=["grayed-out"] if is_disabled else [])
                
                widget.param.watch(self._update_filterconfig, 'value')
                widgets.append(widget)
                
            family_widgets[family] = pn.Column(*widgets, name=family)

        self.accordion_families = pn.Accordion(*[(f"{family}", widget) for family, widget in family_widgets.items()])
        
    def __panel__(self):
        return pn.Card(self.accordion_families, title="🔍\t Filtres de recherche")
    
    def _filter_config(self, df_commands, config_allowed):
        # Filtre le DataFrame en fonction des valeurs définies dans config_allowed
        config_filtered_df = df_commands.copy()
        for col, allowed_values in config_allowed.items():
            if allowed_values:  # S'il y a des valeurs autorisées pour cette colonne
                config_filtered_df = config_filtered_df[config_filtered_df[col].isin(allowed_values)]
        return config_filtered_df.index

    # Callback pour mettre à jour config_allowed et déclencher le filtrage
    def _update_filterconfig(self, event):
        if len(event.old) > 1 :
            self._config_allowed[event.obj.name] = event.new
            #event.obj.param.disabled = len(event.new) == 1 #A tester
            config_filtered = self._filter_config(self.df, self._config_allowed)
            self.config_selector.param.config_options= config_filtered
        else :
            event.obj.param.value = event.old
   


##################################### Niveau 3 : Commande ####################################

###################################################
## Gestion des données niveau 3 sélection unique ##
###################################################

class ConfigUniqueSelector (pn.viewable.Viewer) :
    df = param.DataFrame(doc="Le dataframe contenant les données")
    value = param.String(default= '-', allow_refs=True)
    config_selector = param.ClassSelector(default=None, class_=pn.viewable.Viewer, doc="Widget MultiChoice")
    def __init__ (self, **params):
        super().__init__(**params)
        df = self.df.loc[self.config_selector.value]
        
        self.radio_group = pn.widgets.RadioBoxGroup(
            name='Configurations', 
            options=list(df.Config_Alias) if hasattr(df, 'Config_Alias') else [],
            value=list(df.Config_Alias)[0]  if len(df.Config_Alias) > 0 else '-', 
            inline=False )
        self._update_value()
        self.radio_group.param.watch(self._update_value, "value")

    def __panel__(self):
        return pn.Column(
            pn.pane.Markdown(f"**{self.radio_group.name} :** "),
            self.radio_group)

    def _update_value(self, event = None):
        """
        Met à jour la propriété `value` en fonction de la sélection.
        """
        df = self.df.loc[self.config_selector.value]
        self.value = df['Config_Alias'].get(self.radio_group.value, '-')
    
    @param.depends('config_selector.value', watch=True)   
    def _setLabel(self):
        df = self.df.loc[self.config_selector.value]
        if df.empty:
            # Griser et verrouiller le widget
            self.radio_group.disabled = True
            self.radio_group.value = '-'
            self.radio_group.options = []
        else:
            # Réactiver et mettre à jour les options
            self.radio_group.disabled = False
            self.radio_group.options = list(df.Config_Alias)
            self.radio_group.value = df.Config_Alias.iloc[0] if len(df.Config_Alias) > 0 else '-'

################################################
## Gestion des données niveau 2 avec filtrage ##
################################################

class LogViewer(pn.viewable.Viewer):
    df = param.DataFrame(doc="Le dataframe contenant les données")
    config_selector = param.ClassSelector(default=None, class_=pn.viewable.Viewer, doc="Widget MultiChoice")
    
    def __init__(self, **params):
        super().__init__(**params)
        
        # Initialisation des onglets
        self.output_pane = pn.pane.Markdown("Sélectionnez une configuration pour voir les fichiers.")

        self.radioBoutton = ConfigUniqueSelector(df = self.df, name="One Configuration Selection", config_selector= self.config_selector)
        # Mise à jour des onglets en fonction de la sélection
        self.radioBoutton.param.watch(self._update_tabs, "value")
        # Mettre à jour la sélection dans le panneau

    @param.depends('config_selector.value', watch=True)
    def _update_tabs(self, event = None):
        # Récupérer la ligne sélectionnée
        selected_row = self.df.loc[self.config_selector.value]
        
        # Mettre à jour le contenu des onglets
        if 'log' in self.df.columns:
            self.output_pane.object = f"### Fichier output\n```\n{df['log'].iloc[0]}\n```"
        else:
            self.output_pane.object = "### Fichier output\n```\nAucun log disponible.\n```"

    def __panel__(self):
        # Affichage du sélecteur et des onglets
        return pn.Column(
            self.radioBoutton,
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


#####################################################################################
##################################### Assemblage ####################################
#####################################################################################

###########################################
## Chargement des données depuis Gitlab ###
###########################################

GITLAB_PACKAGE_URL = "https://gitlab.inria.fr/api/v4/projects/1420/packages/generic/elk-export/latest/"

async def load_parquet(name):
    url = GITLAB_PACKAGE_URL + name + ".parquet"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            buf = BytesIO(response.content)

            # Lecture par PyArrow pur
            table = pq.read_table(source=buf)
            return table.to_pandas()

    except Exception as e:
        print(f"❌ Erreur lors du chargement de {name}.parquet : {e}")
        return pd.DataFrame()
       
async def load_data():
    
    df_commands =  await load_parquet('command')
    df_commands.set_index('Command_id', inplace=True)

    df_meta = await load_parquet('meta')
    df_meta.set_index('meta_id', inplace=True)

    df_param = await load_parquet('parameters')
    df_param.set_index('param_id', inplace=True)

    df_tasks = await load_parquet('tasks')
    df_tasks.set_index('RUN_id', inplace=True)

    df_runs = await load_parquet('runs')
    df_runs.set_index('RUN_id', inplace=True)

    df_git = await load_parquet('git')
    df_git.set_index('sha1', inplace=True)
    # df_git['date'] = pd.to_datetime(df_git['date'], utc=True)

    # Générer Config_Alias depuis config
    df_commands['Config_Alias'] = df_commands.index + " : " + df_commands['Command_short'] + "_" + df_commands['sha1']
    config_aliases = dict(zip(df_commands.index, df_commands['Config_Alias']))

    pn.state.cache['db'] =  {
        "commands": df_commands,
        "tasks": df_tasks,
        "runs": df_runs,
        "git": df_git,
        "meta": df_meta,
        "param": df_param,
        "config_aliases": config_aliases
    }

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

# Charger les données initiales
pn.state.onload(load_data)

db = pn.state.cache['db']

#######################
## Paramêtre du site ##
#######################

noise_label ={}
noise_label['Eb/N0'] = 'Signal Noise Ratio(SNR).Eb/N0(dB)'
noise_label['Es/N0'] = 'Signal Noise Ratio(SNR).Es/N0(dB)'
noise_label['Sigma'] = 'Signal Noise Ratio(SNR).Sigma'

noiseScale = NoiseScale(noise_label= noise_label)

paramSite = noiseScale

pn.config.raw_css.append("""
.align-right {
    margin-left: auto;
    display: flex;
    justify-content: flex-end;
}
""")

############################
## Assemblage du panel git ##
############################

git_filter = GitFilterModel(df_git=db['git'])

#ajout du code aux commandes
merged_df = db['commands'].merge(db['param'][['Simulation.Code type (C)']], 
                                left_on='param_id',
                                right_index=True,
                                how='left')
merged_df.rename(columns={'Simulation.Code type (C)': 'code'}, inplace=True)

command_filter = CommandFilterModel(df_commands=merged_df, git_filter=git_filter)

class PanelCommit(pn.viewable.Viewer):
    
    command_filter = param.ClassSelector(default=None, class_=CommandFilterModel, doc="Filtre de commandes")
    git_filter = param.ClassSelector(default=None, class_=GitFilterModel, doc="Filtre Git")
    
    def __init__(self, **params):
        super().__init__(**params)
        # Initialisation du tableau de commandes
        self.date_slider    = DateRangeFilter(git_filter=git_filter)
        # Composants construits
        self.code_selector = CodeSelector(cmd_filter_model=self.command_filter)
        self.table = FilteredTable(filter_model=self.git_filter)
        self.indicators = GitIndicators(df_git=db['git'], df_commands=db['commands'], filter_model=self.git_filter)
        self.perfgraph = PerformanceByCommit(git_filter=self.git_filter, command_filter=self.command_filter)


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
        self.command_table.value = self.command_filter.get_filtered_df()

panelCommit = PanelCommit(command_filter=command_filter, git_filter=git_filter)


##################################### Config ####################################

# Performance par niveau de bruit pour les configurations sélectionnées
def plot_performance_metrics_plotly(configs, noiseScale):
    # Si aucune configuration n'est sélectionnée
    if not configs:
        return pn.pane.Markdown("Veuillez sélectionner au moins une configuration pour afficher les performances.")
    
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

#research_config_filter = Research_config_filter(config_selector = config_selector, df = db['commands'])
config_selector = ConfigPanel(command_filter= command_filter)
#df = db['commands'].join(db['param'], on='param_id', rsuffix='_param')

mi_panel = pn.Column(
    Mutual_information_Panels(
        df = db['runs'],
        index_selecter = config_selector,
        noiseScale =noiseScale
    ),
    scroll=True, height=700
)

# panel des configs
panelConfig = pn.Row(
    #pn.Column(select_all_button, clear_button, config_selector, research_config_filter, width=300),

    pn.Column(
        config_selector,
        TableConfig(df=db['commands'], config_selector=config_selector, meta=False),
        pn.Tabs(
            ('ılıılıılıılıılıılı BER/FER', pn.bind(plot_performance_metrics_plotly, config_selector.param.value, noiseScale.param.value)),
            ('⫘⫘⫘ Mutual information', mi_panel)
        ),
        sizing_mode="stretch_width"
    )
)

##################################### Performance par niveau de SNR ####################################

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
    LogViewer(df=db['commands'], config_selector=config_selector),
    #TableConfig(df=db['commands'], config_selector=config_selector, meta=True),
    # task_Time_Histogramme,
    # plot_latence,
    # plot_debit,
    sizing_mode="stretch_width"
)

##################################### Panel Données ####################################

# Widgets d'affichage des informations
config_count = pn.indicators.Number(name="Configurations en base", value=db['commands'].shape[0] if not db['commands'].empty else 0)


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

#panel de la partie data
panelData = pn.Column(config_count,
                    sizing_mode="stretch_width")

##################################### Tableau de bord ####################################

# Panneaux des performances
panel_Performances = pn.Column(
    sizing_mode="stretch_width"
)

# Layout du tableau de bord avec tout dans une colonne et des arrières-plans différents
dashboard = pn.Column(
    pn.pane.HTML("<div style='font-size: 28px;background-color: #e0e0e0; padding: 10px;line-height : 0px;'><h2> ✏️ Niveau 1 : Evolution par commit </h2></div>"),
    panelCommit,
    pn.pane.HTML("<div style='font-size: 28px;background-color: #e0e0e0; padding: 10px;line-height : 0px;'><h2> ☎️ Niveau 2 : BER / FER </h2></div>"),
    panelConfig,
    pn.pane.HTML("<div style='font-size: 28px;background-color: #e0e0e0; padding: 10px;line-height : 0px;'><h2> ⚙️ Niveau 3 : Analyse à la commande</h2></div>"),
    panel_par_config,
    )

ACCENT = "teal"

styles = {
    "box-shadow": "rgba(50, 50, 93, 0.25) 0px 6px 12px -2px, rgba(0, 0, 0, 0.3) 0px 3px 7px -3px",
    "border-radius": "4px",
    "padding": "10px",
  }


#logo = pn.pane.Image("/home/fchemina/aff3ct.github.io/comit_dashboard/image/2ada77c3-f7d7-40ae-8769-b77cc3791e84.webp")
#logo = pn.pane.Image("https://raw.githubusercontent.com/fCheminadeInria/aff3ct.github.io/refs/heads/master/comit_dashboard/image/2ada77c3-f7d7-40ae-8769-b77cc3791e84.webp")
logo = pn.pane.Image("https://raw.githubusercontent.com/fCheminadeInria/aff3ct.github.io/refs/heads/master/comit_dashboard/image/93988066-1f77-4b42-941f-1d5ef89ddca2.webp")


dashboard= pn.template.FastListTemplate(
    title="Commits Dashboard",
    sidebar=[logo, paramSite,  pn.layout.Divider(), panelData],
    main=[dashboard],
    main_layout=None,
    accent=ACCENT,
    theme_toggle=False,
)


# Lancer le tableau de bord
if args.local :
    dashboard.show()
else :
    dashboard.servable()









