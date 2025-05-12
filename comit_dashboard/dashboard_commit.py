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
import pyarrow.parquet as pq
from io import BytesIO
import httpx

print(ud.unidata_version)

# Initialiser Panel
pn.extension("plotly", sizing_mode="stretch_width")  # Adapter la taille des widgets et graphiques à la largeur de l'écran

# Générer une palette de couleurs
colors = px.colors.qualitative.Plotly

##################################### Component ####################################

class ConfigPanel(pn.viewable.Viewer) :
        # Paramètres configurables
    df = param.DataFrame(doc="Le dataframe contenant les données à filtrer")
    colors = param.Dict(default={},doc="Couleurs des courbes")
    value = param.List(default=[],doc="Liste des index sélectionnées" )

    def __init__(self, **params):
        super().__init__(**params)

        # self.config_filtered = db['commands'].index
        config_options = self.df['Config_Alias'].unique().tolist() if not db['commands'].empty else []
        self.config_selector = pn.widgets.MultiChoice(name="Sélectionnez les configurations", options=config_options)

        # Sélecteur des configs
        self.select_all_button = pn.widgets.Button(name="Tout sélectionner", button_type="success")
        self.clear_button = pn.widgets.Button(name="Tout désélectionner", button_type="warning")

        self.select_all_button.on_click(self.select_all_configs)
        self.clear_button.on_click(self.clear_configs)
#        pn.bind(self._update_value, self.config_selector)
        self.config_selector.param.watch(self._update_value, 'value')



    def __panel__(self):
        return pn.Column(
            self.select_all_button, 
            self.clear_button, 
            self.config_selector, 
            Research_config_filter(config_selector = self.config_selector, df = self.df), 
            width=300)
    
    
    def _update_value(self, event = None):
        # Met à jour `self.value` en fonction des configurations sélectionnées
        selected_configs = self.config_selector.value
        if len(selected_configs) !=0 :
            self.value = self.df[ self.df["Config_Alias"].isin(selected_configs)].index.tolist()
        else :
            self.value = []
            
        # Met à jour les couleurs
        for i, color in enumerate(px.colors.qualitative.Plotly[:len(self.value)]):
            self.colors[self.value[i]] = color
            
    # Boutons d'agrégat
    def select_all_configs(self, event=None):
        self.config_selector.value = self.config_selector.options

    def clear_configs(self, event=None):
        self.config_selector.value = []

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
    #colors = param.ClassSelector(default=None, class_=pn.viewable.Viewer,doc="Couleurs des courbes")

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
            df_filtred = self.df.loc[index] 
        
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
        
        # Générer une palette de couleurs automatiquement selon le nombre de configurations
        if self.lab_group : 
            colors = px.colors.qualitative.Plotly[:len(index) * len(df_filtred[self.lab_group].unique())]
        else :
            colors = px.colors.qualitative.Plotly[:len(index) * len(df_filtred.index.unique())]

        fig = go.Figure()

        # Ajouter une trace pour chaque configuration et tâche
        for i, config in enumerate(index):
            # Filtrer les données pour chaque configuration
            config_data = df_filtred.loc[config]
            alias = db['commands'].loc[config, 'Config_Alias'] #variable global pas propre mais commode
            if self.lab_group :
                for j, t in enumerate(config_data[self.lab_group].unique()):  
                    task_data = config_data[config_data[self.lab_group] == t]
                    snr = task_data[noiseKey]
                    y_values = task_data[self.lab]         
                    
                    
                    if show_envelope :
                        y_values_min = task_data[self.labmin]  
                        y_values_max = task_data[self.labmax]   
                        
                        # Courbe pour la latence avec enveloppe
                        fig.add_trace(go.Scatter(
                            x=snr, y=y_values_max,
                            fill=None, mode='lines+markers',
                            line=dict(width=2, dash='dash', color=colors[i * len(config_data[self.lab_group].unique()) + j]),
                            marker=dict(symbol='x', size=6),
                            showlegend=False
                        ))
                        fig.add_trace(go.Scatter(
                            x=snr, y=y_values_min,
                            fill='tonexty', mode='lines+markers',
                            line=dict(width=2, dash='dash', color=colors[i * len(config_data[self.lab_group].unique()) + j]),
                            marker=dict(symbol='x', size=6),
                            name=f"min/max - {alias} - {t}"  
                        ))

                    
                    fig.add_trace(go.Scatter(
                        x=snr, y=y_values,
                        mode='lines+markers',
                        line=dict(width=2, color=colors[i * len(config_data[self.lab_group].unique()) + j]),
                        name=f"{self.lab} - {alias} - {t}"  
                    ))
            else :
                snr = config_data[noiseKey]
                y_values = config_data[self.lab]         
                
                if show_envelope :
                    y_values_min = config_data[self.labmin]  
                    y_values_max = config_data[self.labmax]   
                    
                    # Courbe pour la latence avec enveloppe
                    fig.add_trace(go.Scatter(
                        x=snr, y=y_values_max,
                        fill=None, mode='lines+markers',
                        line=dict(width=2, dash='dash', color=colors[i]),
                        marker=dict(symbol='x', size=6),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=snr, y=y_values_min,
                        fill='tonexty', mode='lines+markers',
                        line=dict(width=2, dash='dash', color=colors[i]),
                        marker=dict(symbol='x', size=6),
                        name=f"min/max - {config}"  
                    ))

                
                fig.add_trace(go.Scatter(
                    x=snr, y=y_values,
                    mode='lines+markers',
                    line=dict(width=2, color=colors[i]),
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

class Mutual_information_Panels (pn.viewable.Viewer) :
    # Paramètres configurables
    df = param.DataFrame(doc="Le dataframe contenant les données")
    index_selecter = param.ClassSelector(default=None, class_=pn.viewable.Viewer, doc="Widget MultiChoice")
    noiseScale = param.ClassSelector(default=None, class_=pn.viewable.Viewer,doc="Choix de l'échelle de bruit par passage du label de la colonne")

    def __init__(self, **params):
        super().__init__(**params)
        
        self.colors = px.colors.qualitative.Plotly[:len(self.index_selecter.value)]
        
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
            df_filtred = self.df.loc[index] 
        
        # Si pas de données de tâches pour les configurations sélectionnées
        if df_filtred.empty:
            return pn.pane.Markdown(f"Mutual Information : Pas de données complètes d'information mutuelle disponibles pour les configurations sélectionnées.")

        # Générer une palette de couleurs automatiquement selon le nombre de configurations
        fig = go.Figure()
        
        # Ajouter une trace pour chaque configuration et tâche
        for i, config in enumerate(index):
            # Filtrer les données pour chaque configuration
            config_data = df_filtred.loc[config]
            snr = config_data[noiseKey]
            n_trials = config_data["Mutual Information.n_trials"]         
                
            fig.add_trace(go.Scatter(
                x=snr, y=n_trials,
                mode='markers',
                line=dict(width=2, color=colors[i]),
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
        df_commands = df_commands.drop('Config_Alias', axis=1)
        
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
            config_filtered = self._filter_config(db['commands'], self._config_allowed)
            self.config_selector.param.config_options= config_filtered
        else :
            event.obj.param.value = event.old
        
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
        if self.meta :
            filtered_df = self.df.filter(regex=r"^(Meta)\.")
        else :
            filtered_df = self.df.filter(regex=r"^(Simulation)\.")
        
        filtered_df.columns = filtered_df.columns.str.replace(r"^(Meta\.|Simulation\.)", "", regex=True)
        filtered_df = filtered_df.loc[self.config_selector.value]
        return filtered_df


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
            self.radio_group.value = df.Config_Alias[0] if len(df.Config_Alias) > 0 else '-'

class LogViewer(pn.viewable.Viewer):
    df = param.DataFrame(doc="Le dataframe contenant les données")
    config_selector = param.ClassSelector(default=None, class_=pn.viewable.Viewer, doc="Widget MultiChoice")
    
    def __init__(self, **params):
        super().__init__(**params)
        
        # Initialisation des onglets
        self.output_pane = pn.pane.Markdown("Sélectionnez une configuration pour voir les fichiers.")
        self.error_pane = pn.pane.Markdown("")
        #self.report_pane = pn.pane.HTML("")
        
        # Onglets pour afficher les fichiers
        self.tabs = pn.Tabs(
            ("📄 Output", pn.Column(self.output_pane, scroll=True, height=600)),
            ("⚠ Erreur", pn.Column(self.error_pane, scroll=True, height=600)),
            #("📊 Rapport énergétique par Degenerium", pn.Column(self.report_pane, scroll=True, height=600))
        )
        
        self.radioBoutton = ConfigUniqueSelector(df = self.df, name="One Configuration Selection", config_selector= self.config_selector)
        # Mise à jour des onglets en fonction de la sélection
        self.radioBoutton.param.watch(self._update_tabs, "value")
        # Mettre à jour la sélection dans le panneau

    @param.depends('config_selector.value', watch=True)
    def _update_tabs(self, event = None):
        # Récupérer la ligne sélectionnée
        selected_row = self.df.loc[self.config_selector.value]
        
        # Mettre à jour le contenu des onglets
        self.output_pane.object = f"### Fichier output\n```\n{self.read_file(selected_row, 'out')}\n```"
        self.error_pane.object  = f"### Fichier erreur\n```\n{self.read_file(selected_row, 'err')}\n```"
        
        # if args.local :
        #     self.report_pane.object = self.read_file(selected_row, 'html')
        # else :
        #     self.report_pane.object = f"<iframe src='{self.read_file(selected_row, 'html')}' width='100%' height='400'></iframe>"

    def __panel__(self):
        # Affichage du sélecteur et des onglets
        grid = pn.GridSpec(sizing_mode="stretch_width", ncols=4)
        grid[0, 0] = self.radioBoutton
        grid[0, 1:] = self.tabs
        return grid

    def read_file(self, selected_row, ext):
        
        if len(selected_row) == 0 :
            return ""
        try:
            # Accéder à la valeur de 'file_path' dans la série
            if ext != 'html':
                path = 'file_path'
            else :
                path = 'html_degenerium_path'
            
            file_path = selected_row[path] if isinstance(selected_row[path], str) else selected_row[path].values[0]
            
            
            if ext != 'html' or args.local :
                if file_path:
                    # Vérification si file_path existe et est valide
                    with open(args.database_path + file_path + "." + ext, 'r') as file:
                        return file.read()
                else:
                    return f"Le fichier n'a pas de chemin spécifié pour l'extension {ext}."
            else :
                return file_path


        except FileNotFoundError:
            return f"Le fichier {ext} est introuvable."
        except Exception as e:
            return f"Erreur lors de la lecture du fichier {ext}: {str(e)}"
   

##################################### Chargement ####################################

GITLAB_PACKAGE_URL = "https://gitlab.inria.fr/api/v4/projects/1420/packages/generic/elk-export/latest/"

TABLES = ["tasks", "performances", "git", "command", "meta", "runs"]


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

##################################### Panel Données ####################################

# Widgets d'affichage des informations
config_count = pn.indicators.Number(name="Nombre de configurations", value=db['commands'].shape[0] if not db['commands'].empty else 0)
git_version_count = pn.indicators.Number(name="Nombre de commit avec des données", value=db['commands']['sha1'].nunique() if not db['git'].empty else 0)
commit_count = pn.indicators.Number(name="Nombre de commits historisés dans Git", value=db['git'].shape[0] if not db['git'].empty else 0)

# Créer un indicateur pour afficher la date du commit le plus récent
latest_commit_date = db['git']['date'].max() if not db['git'].empty else "Aucune date disponible"
latest_commit_date_str = latest_commit_date.strftime('%Y-%m-%d %H:%M:%S') if latest_commit_date != "Aucune date disponible" else latest_commit_date

# Extraire la date du commit le plus récent
latest_commit_date = db['git']['date'].max() if not db['git'].empty else "Aucune date disponible"

# Créer un widget statique pour afficher la date du commit le plus récent
latest_commit_date_display = pn.Column(
        pn.widgets.StaticText(name="Date du dernier commit",css_classes=["tittle_indicator-text"]),
        pn.widgets.StaticText(value=str(latest_commit_date),css_classes=["indicator-text"])
)

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
                   git_version_count, 
                   commit_count,
                   latest_commit_date_display,
        sizing_mode="stretch_width")

##################################### Paramêtre du site ####################################

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

##################################### Panel Git ####################################

def filter_data(df_git, date_range):
    start_date, end_date = date_range
    # Filtrage par date 
    start_date = datetime.combine(date_range[0], datetime.min.time())
    end_date   = datetime.combine(date_range[1], datetime.min.time())
    
    # Filtrage des données en fonction de la plage de dates
    filtered_df = df_git[(df_git['date'] >= start_date) & (df_git['date'] <= end_date)]   
    
    # Mise à jour de la table avec les données filtrées
    table_commit.value = filtered_df

# Lier le filtre au slider et au RadioButton
def update_filter(event):
    date_range = date_range_slider.value
    filter_data(db['git'], date_range)


# Configuration de l'intervalle de dates pour le DateRangeSlider
min_date = db['git']['date'].min() if not db['git'].empty else datetime(2000, 1, 1)
max_date = db['git']['date'].max() if not db['git'].empty else datetime.now()

# Création du DateRangeSlider
date_range_slider = pn.widgets.DateRangeSlider(
    name="Sélectionnez l'intervalle de dates",
    start=min_date,
    end=max_date,
    value=(min_date, max_date),
)

#table de données Git
table_commit = pn.widgets.DataFrame(db['git'], name='Table de Données', text_align = 'center')

# Lier les événements aux widgets
date_range_slider.param.watch(update_filter, 'value')

# Initialisation de la table avec les données filtrées par défaut
filter_data(db['git'], date_range_slider.value)

panelCommit = pn.Column(
    pn.Column(date_range_slider),
    table_commit,
)

##################################### Config ####################################

# Performance par niveau de bruit pour les configurations sélectionnées
def plot_performance_metrics_plotly(configs, noiseScale):
    # Si aucune configuration n'est sélectionnée
    if not configs:
        return pn.pane.Markdown("Veuillez sélectionner au moins une configuration pour afficher les performances.")
    
    filtered_df_runs = db['runs'].loc[configs]
    if filtered_df_runs.empty:
        return pn.pane.Markdown("Pas de données de performance disponibles pour les configurations sélectionnées.")
    
    fig = go.Figure()
    
    # Générer une palette de couleurs automatiquement selon le nombre de configurations
    colors = px.colors.qualitative.Plotly[:len(configs)]  # Choisir des couleurs depuis Plotly, ajustées à la taille de configs
    
    for i, config in enumerate(configs):
        config_data = filtered_df_runs.loc[config]
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
config_selector = ConfigPanel(df = db['commands'])

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
    config_selector,
    pn.Column(
        TableConfig(df=db['commands'], config_selector=config_selector, meta=False),
        pn.Tabs(
            ('ılıılıılıılıılıılı BER/FER', pn.bind(plot_performance_metrics_plotly, config_selector.param.value, noiseScale.param.value)),
            ('⫘⫘⫘ Mutual information', mi_panel)
        ),
        pn.pane.HTML("<div style='font-size: 20px;background-color: #e0e0e0; padding: 5px;line-height : 0px;'><h2> ✏️ Logs</h2></div>"),
        LogViewer(df=db['commands'], config_selector=config_selector),
        sizing_mode="stretch_width"
    )
    
    
)

##################################### Performance par niveau de SNR ####################################

# Histogramme des temps des jobs
task_Time_Histogramme = Tasks_Histogramme(
    multi_choice_widget = config_selector,
    df = db['tasks'],
    noiseScale = noiseScale
) 

plot_debit = Panel_graph_envelope(
    multi_choice_widget = config_selector,
    df = db['tasks'],
    lab ="Measured throughput Average", 
    labmin="Measured throughput Mininmum", 
    labmax="Measured throughputMaximum", 
    lab_group='Task',
    Ytitle = "Débit",
    noiseScale = noiseScale
)

plot_latence = Panel_graph_envelope(
    multi_choice_widget = config_selector,
    df = db['tasks'],
    lab ="Measured latency Average", 
    labmin="Measured latency Mininmum", 
    labmax="Measured latency Maximum", 
    lab_group='Task',
    Ytitle = "Latence",
    noiseScale = noiseScale
)    
    
panel_level_noise = pn.Column(
    TableConfig(df=db['commands'], config_selector=config_selector, meta=True),
    task_Time_Histogramme,
    plot_latence,
    plot_debit,
    sizing_mode="stretch_width"
)

##################################### Tableau de bord ####################################

# Panneaux des performances
panel_Performances = pn.Column(
    
    sizing_mode="stretch_width"
)

# Layout du tableau de bord avec tout dans une colonne et des arrières-plans différents
dashboard = pn.Column(
    pn.pane.HTML("<div style='font-size: 28px;background-color: #e0e0e0; padding: 10px;line-height : 0px;'><h2> ✏️ Git</h2></div>"),
    panelCommit,
    pn.pane.HTML("<div style='font-size: 28px;background-color: #e0e0e0; padding: 10px;line-height : 0px;'><h2> ☎️ Performances métiers </h2></div>"),
    panelConfig,
    pn.pane.HTML("<div style='font-size: 28px;background-color: #e0e0e0; padding: 10px;line-height : 0px;'><h2> ⚙️ IT performances</h2></div>"),
    panel_level_noise,
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
    sidebar=[logo, panelData, pn.layout.Divider(), paramSite],
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









