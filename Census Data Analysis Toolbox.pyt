# -*- coding: utf-8 -*-
#Author: Ryan Lundin , relundin@iastate.edu
#Date: 11/25/2025
#Purpose: To visualize census tract data within ArcGIS and on a Graph

#Import Modules
import arcpy
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
import numpy as np

#Inital Set up of Toolbox
class Toolbox(object):
    def __init__(self):
        self.label = "Story County Census Choropleth"
        self.alias = "story_census"
        self.tools = [CensusChoroplethTool]


class CensusChoroplethTool(object):
    def __init__(self):
        self.label = "Census Choropleth Map"
        self.description = (
            "Builds a choropleth shapefile, displays it in ArcGIS with "
            "Graduated Colors (Natural Breaks), and creates a matching "
            "matplotlib choropleth PNG."
        )
        self.canRunInBackground = False

    # Setting Parameter Inputs
    def getParameterInfo(self):

        # Parameter 0 - Ask User to Input CSV file
        in_csv = arcpy.Parameter(displayName="Input Census CSV", name="in_csv", datatype="DEFile", parameterType="Required", direction="Input")
        in_csv.filter.list = ["csv"]

        # Paramter 1 - Ask User to Input GeoJSON file
        in_geojson = arcpy.Parameter(displayName="Input GeoJSON (Geography polygons)", name="in_geojson", datatype="DEFile", parameterType="Required", direction="Input")
        in_geojson.filter.list = ["geojson", "json"]

        # Parameter 2 - Ask User to Select and Output Folder for Feature Class
        out_gis_folder = arcpy.Parameter(displayName="Output Folder for ArcGIS Files", name="out_gis_folder", datatype="DEFolder", parameterType="Required", direction="Input")

        # Parameter 3 - Ask the User for attribute name for Feature Class and CSV Join
        attr_name = arcpy.Parameter(displayName="CSV Attribute Name to Plot (column header)", name="attr_name", datatype="GPString", parameterType="Required", direction="Input")

        # Parameter 4 - Drop Down menu for Map Display Options
        arcgis_cmap = arcpy.Parameter(
            displayName="ArcGIS Color Ramp",
            name="arcgis_cmap",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        arcgis_cmap.filter.type = "ValueList"

        # Drop Down Menu options
        arcgis_cmap.filter.list = [
            "Default",
            "Greens",
            "Reds",
            "Blue-Green",
            "Purple-Yellow"
        ]
        arcgis_cmap.value = "Default"

        # Paramter 5 - Graph display options
        mpl_cmap = arcpy.Parameter(
            displayName="Matplotlib Colormap",
            name="mpl_cmap",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        mpl_cmap.filter.type = "ValueList"
        mpl_cmap.filter.list = [
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "Greys",
            "hot"
        ]
        mpl_cmap.value = "viridis"

        # Parameter 6 - Number of classes for symbology of map and graph
        n_bins = arcpy.Parameter(
            displayName="Number of Classes (value bins)",
            name="n_bins",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        n_bins.value = 5
        n_bins.filter.type = "Range"
        n_bins.filter.list = [2, 9]  

        # Parameter 7 - Ask the user to choose a folder to save the graph png
        out_png_folder = arcpy.Parameter(
            displayName="Output Folder for Choropleth PNG",
            name="out_png_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )

        return [
            in_csv, in_geojson, out_gis_folder,
            attr_name,
            arcgis_cmap, mpl_cmap, n_bins,
            out_png_folder
        ]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    # Running the Tool 
    def execute(self, parameters, messages):
        arcpy.env.overwriteOutput = True

        # Bring up parameters
        in_csv = parameters[0].valueAsText
        in_geojson = parameters[1].valueAsText
        out_gis_folder = parameters[2].valueAsText
        attr_name = parameters[3].valueAsText
        arcgis_cmap_choice = parameters[4].valueAsText
        mpl_cmap_choice = parameters[5].valueAsText
        n_bins = int(parameters[6].value)
        out_png_folder = parameters[7].valueAsText

        # Set up the PNG ouput path with a standard name
        out_png = os.path.join(out_png_folder, "graph_output.png")

        arcpy.AddMessage(f"Input CSV: {in_csv}")
        arcpy.AddMessage(f"Input GeoJSON: {in_geojson}")
        arcpy.AddMessage(f"ArcGIS output folder: {out_gis_folder}")
        arcpy.AddMessage(f"Attribute to map: {attr_name}")
        arcpy.AddMessage(f"ArcGIS color ramp: {arcgis_cmap_choice}")
        arcpy.AddMessage(f"Matplotlib colormap: {mpl_cmap_choice}")
        arcpy.AddMessage(f"Number of classes: {n_bins}")
        arcpy.AddMessage(f"Output PNG (graph_output.png): {out_png}")

        # Ensure ArcGIS output folder exists
        if not os.path.exists(out_gis_folder):
            arcpy.AddMessage(
                f"ArcGIS output folder does not exist, creating: {out_gis_folder}"
            )
            os.makedirs(out_gis_folder)

        # Import the selected output for feature class
        shp_path = os.path.join(out_gis_folder, "StoryCensus_Choropleth.shp")
        arcpy.AddMessage(f"Choropleth shapefile will be written to: {shp_path}")

        # Delete existing shapefile
        if arcpy.Exists(shp_path):
            arcpy.AddMessage("Existing choropleth shapefile found; deleting...")
            arcpy.management.Delete(shp_path)

        # Bring in the CSV file and measured paramter
        try:
            import pandas as pd
        except ImportError:
            raise arcpy.ExecuteError(
                "pandas is required. Install it in the ArcGIS Pro Python environment."
            )

        if not os.path.exists(in_csv):
            raise arcpy.ExecuteError(f"CSV file not found: {in_csv}")

        df = pd.read_csv(in_csv)

        if "Geography" not in df.columns:
            raise arcpy.ExecuteError("The CSV must contain a 'Geography' column.")

        if attr_name not in df.columns:
            raise arcpy.ExecuteError(
                f"Attribute '{attr_name}' not found. "
                f"Columns: {list(df.columns)}"
            )

        raw_series = df[attr_name].astype(str)

        
        contains_pct = raw_series.str.contains("%").mean() > 0.3
        name_has_pct = ("%" in attr_name) or ("percent" in attr_name.lower())
        is_percent = contains_pct or name_has_pct

        cleaned_series = (
            raw_series
            .str.replace("%", "", regex=False)
            .str.replace("±", "", regex=False)
            .str.replace(" ", "", regex=False)
        )
        numeric_values = pd.to_numeric(cleaned_series, errors="coerce")

        name_to_value = {}
        for geo_name, val in zip(df["Geography"].astype(str), numeric_values):
            if not np.isnan(val):
                name_to_value[geo_name] = float(val)

        if not name_to_value:
            raise arcpy.ExecuteError(
                "No valid numeric values found for the selected attribute."
            )

        arcpy.AddMessage(
            f"Found {len(name_to_value)} geographies with numeric values."
        )

        # Converting the GeoJSON to a feature class and completing the CSV join connection 
        if not os.path.exists(in_geojson):
            raise arcpy.ExecuteError(f"GeoJSON file not found: {in_geojson}")

        
        geojson_fc = r"in_memory\geojson_fc"
        csv_table = r"in_memory\csv_table"

        arcpy.AddMessage("Converting GeoJSON to in-memory feature class...")
        if arcpy.Exists(geojson_fc):
            arcpy.management.Delete(geojson_fc)
        arcpy.conversion.JSONToFeatures(in_geojson, geojson_fc)

        arcpy.AddMessage("Converting CSV to in-memory table...")
        if arcpy.Exists(csv_table):
            arcpy.management.Delete(csv_table)
        arcpy.conversion.TableToTable(in_csv, "in_memory", "csv_table")

        # Join the actual CSV to the GeoJSON
        arcpy.AddMessage(
            f"Joining CSV table to features on NAME (features) = Geography (CSV) "
            f"for attribute '{attr_name}'..."
        )
        arcpy.management.JoinField(
            in_data=geojson_fc,
            in_field="NAME",
            join_table=csv_table,
            join_field="Geography",
            fields=[attr_name]
        )

        arcpy.AddMessage("Ensuring 'Value' field exists on joined feature class...")
        field_names = [f.name for f in arcpy.ListFields(geojson_fc)]
        if "Value" not in field_names:
            arcpy.management.AddField(geojson_fc, "Value", "DOUBLE")

        arcpy.AddMessage("Populating 'Value' field from cleaned CSV data...")
        with arcpy.da.UpdateCursor(geojson_fc, ["NAME", "Value"]) as cursor:
            for row in cursor:
                geo_name = str(row[0])
                if geo_name in name_to_value:
                    row[1] = name_to_value[geo_name]
                else:
                    row[1] = None
                cursor.updateRow(row)

        arcpy.AddMessage("Joined feature class attribute population complete.")

        # Create the feature class from the joined GeoJSON and CSV data
        arcpy.AddMessage(
            "Copying joined feature class with 'Value' field to final shapefile..."
        )
        arcpy.management.CopyFeatures(geojson_fc, shp_path)

        fc_for_mapping = shp_path

        # Setting the symbology of the arcgis map 
        edges = None  
        try:
            aprx = arcpy.mp.ArcGISProject("CURRENT")
            m = aprx.activeMap
            if m is None:
                maps = aprx.listMaps()
                if maps:
                    m = maps[0]

            if m is None:
                arcpy.AddWarning(
                    "No active map found. Layer will not be added to a map."
                )
            else:
                arcpy.AddMessage("Adding choropleth layer to current map...")
                lyr = m.addDataFromPath(fc_for_mapping)

                
                lyr.name = f"Choropleth - {attr_name}"

               
                sym = lyr.symbology
                sym.updateRenderer("GraduatedColorsRenderer")
                renderer = sym.renderer

                renderer.classificationField = "Value"
                renderer.normalizationField = None
                renderer.breakCount = n_bins
                renderer.classificationMethod = "NaturalBreaks"

                
                if arcgis_cmap_choice != "Default":
                    ramp_pattern = self._arcgis_ramp_pattern(arcgis_cmap_choice)
                    ramps = aprx.listColorRamps(ramp_pattern)
                    if ramps:
                        renderer.colorRamp = ramps[0]
                        arcpy.AddMessage(
                            f"Using ArcGIS color ramp matching '{ramp_pattern}'."
                        )
                    else:
                        arcpy.AddWarning(
                            f"No color ramp matching '{ramp_pattern}' found. "
                            "Using default ArcGIS color ramp."
                        )

                lyr.symbology = sym  

                
                lyr.showLabels = True
                lbl_classes = lyr.listLabelClasses()
                if lbl_classes:
                    lbl = lbl_classes[0]
                    lbl.showClassLabels = True
                    lbl.expression = "$feature.Value"
                else:
                    arcpy.AddWarning("Could not find label classes to configure labels.")

                arcpy.AddMessage(
                    "ArcGIS symbology applied: Graduated Colors, Natural Breaks, "
                    f"{n_bins} classes on 'Value'. Labels enabled."
                )

                sym2 = lyr.symbology
                renderer2 = sym2.renderer

                lb = renderer2.lowerBound
                brks = renderer2.classBreaks  
                edges = [lb] + [b.upperBound for b in brks]

                arcpy.AddMessage("Class breaks from ArcGIS Natural Breaks:")
                for i in range(n_bins):
                    arcpy.AddMessage(
                        f"Class {i+1}: {edges[i]:.3f} to {edges[i+1]:.3f}"
                    )

        except Exception as e:
            arcpy.AddWarning(
                f"Could not add layer to map or set symbology automatically: {e}"
            )

        
        if edges is None:
            arcpy.AddMessage(
                "Falling back to equal-interval classification for matplotlib."
            )
            all_vals = np.array(list(name_to_value.values()))
            vmin = float(np.min(all_vals))
            vmax = float(np.max(all_vals))
            if vmin == vmax:
                vmin -= 0.5
                vmax += 0.5
            edges = np.linspace(vmin, vmax, n_bins + 1)

        # Create the general matplotlib graph by using the selected symbology
        arcpy.AddMessage("Building matplotlib choropleth...")

        def classify_value(val, edge_list):
            for i in range(n_bins):
                low = edge_list[i]
                high = edge_list[i + 1]
                if i < n_bins - 1:
                    if low <= val < high:
                        return i + 1
                else:
                    if low <= val <= high:
                        return i + 1
            return None

        
        with open(in_geojson, "r") as f:
            gj = json.load(f)

        if "features" not in gj:
            raise arcpy.ExecuteError(
                "GeoJSON does not contain a 'features' collection."
            )

        polygons = []
        poly_classes = []

        def add_polygon_from_coords(coords, cls_index):
          
            if isinstance(coords[0][0][0], (list, tuple)):
                for poly in coords:
                    outer = poly[0]
                    arr = np.array(outer)
                    polygons.append(arr)
                    poly_classes.append(cls_index)
            else:
                
                outer = coords[0]
                arr = np.array(outer)
                polygons.append(arr)
                poly_classes.append(cls_index)

        for feat in gj["features"]:
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})

            geo_name = str(props.get("NAME", ""))
            if geo_name not in name_to_value:
                continue

            val = name_to_value[geo_name]
            cls = classify_value(val, edges)
            if cls is None:
                continue

            geom_type = geom.get("type", "")
            coords = geom.get("coordinates", None)
            if not coords:
                continue

            if geom_type in ("Polygon", "MultiPolygon"):
                add_polygon_from_coords(coords, cls)

        if not polygons:
            raise arcpy.ExecuteError(
                "No polygons with valid classes found for matplotlib plotting."
            )

        
        try:
            base_cmap = getattr(matplotlib.cm, mpl_cmap_choice)
        except AttributeError:
            base_cmap = matplotlib.cm.viridis
            arcpy.AddWarning(
                f"Matplotlib colormap '{mpl_cmap_choice}' not found. Using 'viridis'."
            )

       
        color_positions = np.linspace(0.1, 0.9, n_bins)
        class_colors = [base_cmap(pos) for pos in color_positions]
        cls_to_color = {i + 1: class_colors[i] for i in range(n_bins)}

        face_colors = [cls_to_color[c] for c in poly_classes]
        poly_collection = PolyCollection(
            polygons,
            facecolors=face_colors,
            edgecolors="black",
            linewidths=0.4
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.add_collection(poly_collection)

        all_x = np.concatenate([p[:, 0] for p in polygons])
        all_y = np.concatenate([p[:, 1] for p in polygons])
        ax.set_xlim(all_x.min(), all_x.max())
        ax.set_ylim(all_y.min(), all_y.max())
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(
            f"Choropleth of '{attr_name}' "
            f"({n_bins}-Class Natural Breaks)"
        )

        
        legend_handles = []
        for i in range(n_bins):
            low = edges[i]
            high = edges[i + 1]
            if is_percent:
                label = f"{low:.2f}% – {high:.2f}%"
            else:
                label = f"{low:.2f} – {high:.2f}"
            patch = Patch(
                facecolor=cls_to_color[i + 1],
                edgecolor="black",
                label=label
            )
            legend_handles.append(patch)

        ax.legend(
            handles=legend_handles,
            title="Value Ranges",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.
        )

        plt.tight_layout()

       
        out_dir = os.path.dirname(out_png)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

       
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

        arcpy.AddMessage("Matplotlib choropleth PNG created successfully.")
        arcpy.AddMessage(f"PNG saved to: {out_png}")
        arcpy.AddMessage(
            "Done. The shapefile 'StoryCensus_Choropleth.shp' is in the folder "
            "you selected for ArcGIS outputs and should now be visible in your map."
        )

    
    def _arcgis_ramp_pattern(self, choice):
        """
        Map logical ArcGIS colormap choices to listColorRamps patterns.
        These are approximate and depend on styles in the project.
        """
        if choice == "Greens":
            return "*Green*"
        if choice == "Reds":
            return "*Red*"
        if choice == "Blue-Green":
            return "*Blue*Green*"
        if choice == "Purple-Yellow":
            return "*Purple*Yellow*"
        # Default
        return "*"
