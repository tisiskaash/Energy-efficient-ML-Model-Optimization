import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns

# Find the folder where this Python script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# The CSV files are stored inside the "results" folder
results_dir = os.path.join(script_dir, "results")

# List of result files we want to combine
csv_files = [
    "car_evaluation_results.csv",
    "Dry_Bean_Dataset_results.csv",
    "heart_results.csv",
    "spambase_results.csv",
    "UCI_Credit_Card_results.csv",]

# Create the full path for each CSV file
csv_paths = [os.path.join(results_dir, f) for f in csv_files]

# Check that all required files exist before trying to read them
missing = [f for f in csv_paths if not os.path.exists(f)]

if missing:
    raise FileNotFoundError(f"Missing files: {missing}\n" f"Run `ls {results_dir}` to check what's actually there.")

# Read all CSV files into separate dataframes
all_dfs = [pd.read_csv(f) for f in csv_paths]
# Combine all datasets into one large dataframe
df_all = pd.concat(all_dfs, ignore_index=True)
# Print some basic information about the combined data
print("Datasets included:", df_all["Dataset"].unique().tolist())
print("Rows per dataset:")
print(df_all["Dataset"].value_counts())
print("Total rows:", len(df_all))

# Save the combined data so it can be used by the plotting section
df_all.to_csv(os.path.join(script_dir, "final_data.csv"), index=False)
print("\nSaved combined data to", os.path.join(script_dir, "final_data.csv"))

# Set a clean style for all visual
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams["grid.color"] = "#A0A0A0"
plt.rcParams["grid.alpha"] = 0.3

# Make the edges of chart elements dark grey
plt.rcParams["axes.edgecolor"] = "#444"

# Set up folders and file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "final_data.csv")
visual_dir = os.path.join(script_dir, "visual")

# Create the visual folder if it does not already exist
os.makedirs(visual_dir, exist_ok=True)


# Function to save each chart
def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(visual_dir, f"{name}.png"), dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

# Load and prepare the data
df = pd.read_csv(data_path)
# Mark models whose name starts with "Optimised"
df["Optimised"] = df["Model"].str.startswith("Optimised")
# Remove "Optimised " from the model name to compare the same model before and after optimisation
df["Base_Model"] = df["Model"].str.replace("Optimised ", "", regex=False)

dataset_info = {
    "heart": {"rows": 918, "features": 20, "majority_class_pct": 55.34},
    "spambase": {"rows": 4601, "features": 57, "majority_class_pct": 60.6},
    "car_evaluation": {"rows": 1728, "features": 21, "majority_class_pct": 70.02},
    "Dry_Bean_Dataset": {"rows": 13611, "features": 16, "majority_class_pct": 26.05},
    "UCI_Credit_Card": {"rows": 30000,"features": 23, "majority_class_pct": 77.88}}
info_df = (pd.DataFrame(dataset_info).T.reset_index().rename(columns={"index": "Dataset"}))
df = df.merge(info_df, on="Dataset")
# To mask possible anomalies
ANOMALY_MASK = pd.Series(False, index=df.index)
df_clean = df[~ANOMALY_MASK].copy()

# Calculate model efficiency
df["Efficiency"] = (df["Accuracy"] / df["Energy_Joules"].replace(0, np.nan))
df_clean["Efficiency"] = (df_clean["Accuracy"] / df_clean["Energy_Joules"])

# Match baseline and optimised models
base = (df_clean[~df_clean["Optimised"]].set_index(["Dataset", "Base_Model"]))
opt = (df_clean[df_clean["Optimised"]].set_index(["Dataset", "Base_Model"]))
merged = base[["Accuracy", "Energy_Joules"]].join(opt[["Accuracy", "Energy_Joules"]],lsuffix="_base", rsuffix="_opt")

# Calculate how much accuracy changed.
merged["Acc_Delta_pp"] = (merged["Accuracy_opt"] - merged["Accuracy_base"]) * 100
# Calculate the percentage change in energy use
merged["Energy_Pct_Change"] = ((merged["Energy_Joules_opt"] - merged["Energy_Joules_base"]) / merged["Energy_Joules_base"]) * 100
# Calculate efficiency for the baseline model
merged["Eff_base"] = (merged["Accuracy_base"] / merged["Energy_Joules_base"])
# Calculate efficiency for the optimised model
merged["Eff_opt"] = (merged["Accuracy_opt"] / merged["Energy_Joules_opt"])
merged["Eff_Ratio"] = (merged["Eff_opt"] / merged["Eff_base"])

models = df["Base_Model"].unique()
datasets = df["Dataset"].unique()

# Create colours for the models
palette = sns.color_palette("husl", len(models))
print("Datasets:", list(datasets))
print("Models:", list(models))
print(f"Anomalies excluded from energy-scale visual: "f"{ANOMALY_MASK.sum()} rows")

# 1 - SVM accuracy compared with the majority-class baseline
svm = (df[df["Base_Model"] == "SVM"].pivot(index="Dataset", columns="Optimised", values="Accuracy"))
svm.columns = ["Baseline", "Optimised"]
svm = svm.merge(info_df.set_index("Dataset")["majority_class_pct"] / 100, left_index=True, right_index=True)
svm.columns = ["Baseline", "Optimised", "Majority_Class_Baseline"]
svm = svm.sort_values("Optimised")
# Create the chart
fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(svm))
width = 0.25

# Draw the three bars for each dataset
ax.bar( x - width, svm["Baseline"], width, label="SVM baseline", color="#CF8A1B")
ax.bar(x, svm["Optimised"], width, label="SVM optimised", color="#082446")
ax.bar(x + width, svm["Majority_Class_Baseline"], width, label="Always guess majority class", color="#D492F7A3", alpha=0.7)

# Add dataset names to the x-axis
ax.set_xticks(x)
ax.set_xticklabels(svm.index, rotation=15)
ax.set_ylabel("Accuracy")
ax.set_title("SVM optimisation vs SVM Baseline vs Always guess majority class")
ax.legend()
sns.despine()
# Save the chart
save(fig, "svm_vs_majority_baseline")

# 2 - Energy used for every 1,000 rows

b = df_clean[~df_clean["Optimised"]].copy()
b["Energy_per_1k_rows"] = (b["Energy_Joules"] / (b["rows"] / 1000))
pivot = b.pivot(index="Base_Model",columns="Dataset", values="Energy_per_1k_rows")
pivot = pivot[sorted(pivot.columns, key=lambda c: dataset_info[c]["rows"])]
# Create the heatmap
fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(np.log10(pivot),annot=pivot.round(3), fmt="", cmap="RdYlGn_r", cbar_kws={"label": "log10(J per 1k rows)"}, linewidths=1, linecolor="white", ax=ax)

ax.set_title("Per-row energy cost (ordered by size, left to right)")
ax.set_ylabel("")
save(fig, "energy_per_row_heatmap")

# 3 - Average model ranking

opt_only = df_clean[df_clean["Optimised"]].copy()
# Rank models within each dataset
opt_only["Rank"] = (opt_only.groupby("Dataset")["Accuracy"].rank(ascending=False))
# Calculate each model's average rank across all datasets
avg_rank = (opt_only.groupby("Base_Model")["Rank"].mean().sort_values())
# Create a bar chart
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(avg_rank.index, avg_rank.values, color=
       "#F8FC87FF", edgecolor="black")
ax.set_ylabel("Average accuracy rank across datasets (1 = best)")
ax.set_xlabel("Model")
sns.despine()
save(fig, "average_rank")

# 4 - Efficiency change caused by optimisation

# Calculate the average efficiency improvement for each model
eff = (merged.groupby(level=1)["Eff_Ratio"].mean().sort_values())
# Create the chart
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(eff.index,eff.values,color="#8172B2",edgecolor="white")
# A value of 1 means there was no efficiency change
ax.axvline(1,color="Black", lw=1, linestyle="--", label="No change")
# Add the efficiency value to each bar
for i, v in enumerate(eff.values):
    ax.text(v + 0.1, i, f"{v:.2f}x", va="center", fontsize=9.5)
ax.set_xlabel("Efficiency multiplier " "(optimised accuracy/J ÷ baseline accuracy/J)")
ax.set_title("Efficiency gain from optimisation, by model")
ax.legend()
sns.despine()
save(fig, "efficiency_ratio")

# 5 - Accuracy comparison for all datasets

plot_df = df_clean.copy()
# Convert accuracy from a decimal to a percentage
plot_df["Accuracy_pct"] = plot_df["Accuracy"] * 100
# Create one small chart for each dataset
g = sns.catplot(data=plot_df, x="Base_Model", y="Accuracy_pct", hue="Optimised", col="Dataset", kind="bar", col_wrap=3,
    height=4.5, aspect=1.3, palette=["#051707", "#2FAC17"], sharex=False, sharey=False)
g.set_xticklabels(rotation=45, ha="right", fontsize=9)
g.set_titles("{col_name}")
for ax in g.axes.flat:
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
g.fig.suptitle("Accuracy Comparision: baseline vs optimised", y=1.02, fontsize=14)
# Add extra spacing between rows/columns so multi-word model names don't overlap
g.fig.subplots_adjust(hspace=0.6, wspace=0.3)
# Save the complete set of visual
g.savefig(os.path.join(visual_dir, "accuracy_comparision.png"), dpi=200, bbox_inches="tight", pad_inches=0.15)

# 6 -  Show the result of optimisation

# optimised results
outcome = (merged.dropna(subset=["Acc_Delta_pp", "Energy_Pct_Change"]).reset_index().copy())
# Excluded anomalies
excluded_pairs = set(map(tuple, df.loc[ANOMALY_MASK, ["Dataset", "Base_Model"]].drop_duplicates().values))

def get_outcome(row):
    acc_up = row["Acc_Delta_pp"] > 0
    energy_down = row["Energy_Pct_Change"] < 0
    # Both accuracy and energy improved
    if acc_up and energy_down:
        return "Win-Win"
    # Accuracy improved, but energy did not decrease
    if acc_up and not energy_down:
        return "Accuracy gain"
    # Energy decreased, but accuracy did not improve
    if not acc_up and energy_down:
        return "Energy saving"
    # Both measures became worse
    return "Trade-off loss"

# Add outcome label
outcome["Outcome"] = outcome.apply(get_outcome, axis=1)
# Colours used for the different outcomes
outcome_colors = {"Win-Win": "#20A464", "Accuracy gain": "#3B82C4", "Energy saving": "#E5A52F", "Trade-off loss": "#D9534F"}
model_order = list(models)
dataset_order = list(datasets)
# Create the optimisation outcome matrix
fig, ax = plt.subplots(figsize=(14, 10))
# Go through every model and dataset combination
for row_idx, model in enumerate(model_order):
    for col_idx, dataset in enumerate(dataset_order):
        if (dataset, model) in excluded_pairs:
            ax.add_patch(Rectangle((col_idx, row_idx),1, 1, facecolor="#DDDDDD", edgecolor="white", linewidth=3, hatch="////"))
            ax.text(col_idx + 0.5, row_idx + 0.5, "Excluded\n(anomaly)", ha="center", va="center", fontsize=9, fontweight="bold", color="#666666")
            continue

        # Find the result for this model and dataset
        match = outcome[(outcome["Base_Model"] == model) & (outcome["Dataset"] == dataset)]
        # Skip if no matching result exists
        if match.empty:
            continue
        
        row = match.iloc[0]
        # Draw a coloured square for the result
        ax.add_patch(Rectangle((col_idx, row_idx), 1, 1, facecolor=outcome_colors[row["Outcome"]], edgecolor="white", linewidth=3))
        # Show the accuracy change
        ax.text(col_idx + 0.5, row_idx + 0.43, f"Accuracy\n{row['Acc_Delta_pp']:+.1f} pp", ha="center", va="center", fontsize=10,
                 fontweight="bold", color="white")
        # Show the energy change
        ax.text(col_idx + 0.5, row_idx + 0.75, f"Energy {row['Energy_Pct_Change']:+.0f}%", ha="center", va="center", fontsize=9,
                color="white", alpha=0.95)

# Set the size of the matrix
ax.set_xlim(0, len(dataset_order))
ax.set_ylim(0, len(model_order))
ax.set_xticks(np.arange(len(dataset_order)) + 0.5)
ax.set_yticks(np.arange(len(model_order)) + 0.5)
ax.set_xticklabels([d.replace("_", " ") for d in dataset_order], rotation=25, ha="right", fontsize=10)
ax.set_yticklabels(model_order, fontsize=10)
# Put the first model at the top
ax.invert_yaxis()
ax.set_xlabel("Dataset", fontsize=11, labelpad=10)
ax.set_ylabel("Model", fontsize=11, labelpad=10)
ax.tick_params(length=0)
# Remove the chart border
for spine in ax.spines.values():
    spine.set_visible(False)
# Create the legend
legend_handles = [Line2D([0], [0], marker="s", linestyle="", markersize=13, markerfacecolor=c, markeredgecolor="white", label=l)
                    for l, c in outcome_colors.items()]
ax.legend(handles=legend_handles, title="Optimisation outcome", loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False, fontsize=10)
ax.set_title("Optimisation Outcomes Across Models and Datasets", fontsize=16, fontweight="bold", pad=20)
# Explanation underneath the chart
fig.text(0.5, -0.32, "Accuracy = change in percentage points  •  " "Energy = percentage change from baseline  •  "
            "Negative energy values indicate energy savings", ha="center", fontsize=9.5, transform=ax.transAxes)
plt.tight_layout()
fig.savefig(os.path.join(visual_dir,"optimisation_outcome_matrix.png"),dpi=200, bbox_inches="tight", pad_inches=0.15)

# 7 - Neural Network energy comparison

nn = (df[df["Base_Model"] == "Neural Network"].pivot(index="Dataset", columns="Optimised", values="Energy_Joules"))
# Rename the columns
nn.columns = ["Baseline", "Optimised"]
# Order datasets from smallest to largest
nn = nn.reindex(sorted(nn.index, key=lambda d: dataset_info[d]["rows"]))
fig, ax = plt.subplots(figsize=(9, 5.5))
x_pos = np.arange(len(nn))
# Plot baseline energy
ax.plot(x_pos, nn["Baseline"], "o-", color="#B0B0B0", label="Baseline", markersize=10, lw=2)
# Plot optimised energy
ax.plot(x_pos, nn["Optimised"], "o-", color="#2A78D6", label="Optimised", markersize=10, lw=2)
# Use a log scale because energy values can vary a lot
ax.set_yscale("log")
ax.set_xticks(x_pos)
ax.set_xticklabels(nn.index)
ax.set_ylabel("Energy (Joules, log scale)")
ax.set_title("Neural Network energy trend by dataset size — " "optimisation cuts cost at every scale")
ax.legend()
sns.despine()
save(fig, "nn_trend_line")

# 8 - KNN energy comparison

knn = (df[df["Base_Model"] == "KNN"].pivot(index="Dataset", columns="Optimised", values="Energy_Joules"))
knn.columns = ["Baseline", "Optimised"]
# Create one horizontal line for each dataset
fig, ax = plt.subplots(figsize=(9, 5.5))
y_pos = np.arange(len(knn))
# Draw lines and points for the baseline energy
ax.hlines(y_pos, 0, knn["Baseline"], color="#B0B0B0", lw=2)
ax.scatter(knn["Baseline"], y_pos, color="#B0B0B0", s=130, zorder=3, label="Baseline", edgecolor="white")
# Draw lines and points for the optimised energy
ax.hlines(y_pos + 0.15, 0, knn["Optimised"], color="#2A78D6", lw=2)
ax.scatter(knn["Optimised"], y_pos + 0.15, color="#2A78D6", s=130, zorder=3, label="Optimised", edgecolor="white")
ax.set_yticks(y_pos + 0.075)
ax.set_yticklabels(knn.index)
ax.set_xlabel("Energy (Joules)")
ax.set_title("KNN energy by dataset")
ax.legend()
sns.despine()
save(fig, "knn_lollipop")

# 9 - Overall summary of optimisation results

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Calculate the average accuracy change for each model
# across all datasets
stats_final = (merged.groupby(level=1)["Acc_Delta_pp"].agg(["mean", "std"]).sort_values("mean"))
colors1 = ["#1BAF7A" if v >= 0 else "#D6532A" for v in stats_final["mean"]]
# Draw the accuracy chart
axes[0].barh(stats_final.index, stats_final["mean"], xerr=stats_final["std"], color=colors1, edgecolor="white", capsize=4)
axes[0].axvline(0,color="#444",lw=1)
axes[0].set_xlabel("Mean accuracy change (pp)")
axes[0].set_title("Accuracy impact")

# Calculate the average energy change for each model
stats_final_e = (merged.groupby(level=1)["Energy_Pct_Change"].agg(["mean", "std"]).sort_values("mean"))
colors2 = ["#1BAF7A" if v < 0 else "#DE4416" for v in stats_final_e["mean"]]
axes[1].barh(stats_final_e.index, stats_final_e["mean"], xerr=stats_final_e["std"], color=colors2, edgecolor="white", capsize=4)
axes[1].axvline(0, color="#444", lw=1)
axes[1].set_xlabel("Mean energy change (%)")
axes[1].set_title("Energy impact")
fig.suptitle("NSGA-II optimisation impact, averaged across 5 datasets", fontsize=13, y=1.05)
sns.despine()
save(fig, "summary_two_panel")

model_colors = dict(zip(models, palette))


# 10 - Accuracy vs Energy tradeoff 3
plot_data = merged.reset_index().dropna(subset=["Accuracy_base", "Accuracy_opt", "Energy_Joules_base", "Energy_Joules_opt"])
n_datasets = len(datasets)
ncols = 3
nrows = int(np.ceil(n_datasets / ncols))

BG = "#0D0D0D"
FG = "#EAEAEA"
GRID = "#3A3A3A"

fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows), facecolor=BG)
axes = np.array(axes).reshape(-1)

for i, dataset in enumerate(datasets):

    ax = axes[i]
    ax.set_facecolor(BG)
    ax.grid(False)
    sub = plot_data[plot_data["Dataset"] == dataset]

    if sub.empty:
        ax.axis("off")
        continue

    x_all = pd.concat([sub["Accuracy_base"], sub["Accuracy_opt"]]) * 100
    y_all = pd.concat([sub["Energy_Joules_base"], sub["Energy_Joules_opt"]])
    x_pad = (x_all.max() - x_all.min()) * 0.15 + 1
    y_pad = (y_all.max() / y_all.min()) ** 0.15
    xlim = (x_all.min() - x_pad, x_all.max() + x_pad)
    ylim = (y_all.min() / y_pad, y_all.max() * y_pad)

    for _, row in sub.iterrows():
        color = model_colors[row["Base_Model"]]
        x0, x1 = row["Accuracy_base"] * 100, row["Accuracy_opt"] * 100
        y0, y1 = row["Energy_Joules_base"], row["Energy_Joules_opt"]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, alpha=0.95,
                                                        mutation_scale=18, shrinkA=6, shrinkB=6, connectionstyle="arc3,rad=0.08"), zorder=3)
        ax.scatter(x0, y0, color=color, s=70, marker="o", edgecolor=BG, linewidth=1.2, zorder=4)
        ax.scatter(x1, y1, color=color, s=110, marker="D", edgecolor=BG, linewidth=1.2, zorder=4)
    ax.set_yscale("log")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(dataset.replace("_", " "), fontsize=12, fontweight="bold", color=FG)
    ax.set_xlabel("Accuracy (%)", fontsize=9, color=FG)
    ax.set_ylabel("Energy (Joules, log scale)", fontsize=9, color=FG)
    ax.tick_params(colors=FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

for j in range(n_datasets, len(axes)):
    axes[j].axis("off")

shape_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="grey", markeredgecolor=BG, markersize=8, label="Baseline"),
                plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="grey", markeredgecolor=BG, markersize=9, label="Optimised"),]
model_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=model_colors[m], markeredgecolor=BG, markersize=8, label=m) for m in models]

legend1 = fig.legend(handles=shape_handles, loc="upper center", bbox_to_anchor=(0.27, -0.02),ncol=1, 
                    frameon=False, fontsize=10, labelcolor=FG, title="Marker", title_fontsize=10)
legend1.get_title().set_color(FG)
fig.add_artist(legend1)

legend2 = fig.legend(handles=model_handles, loc="upper center", bbox_to_anchor=(0.65, -0.02), ncol=min(len(models), 4), 
                    frameon=False, fontsize=10, labelcolor=FG, title="Model", title_fontsize=10)
legend2.get_title().set_color(FG)
plt.tight_layout(rect=[0, 0.08, 1, 0.96])
save(fig, "accuracy_energy_tradeoff")


# 11 - Total energy footprint by dataset, stacked by model
footprint = df_clean.groupby(["Dataset", "Base_Model"])["Energy_Joules"].sum().unstack(fill_value=0)
footprint = footprint.loc[sorted(footprint.index, key=lambda d: dataset_info[d]["rows"])]
footprint = footprint[list(models)]

BG = "#0D0D0D"
FG = "#EAEAEA"
GRID = "#7DFF1239"

fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
ax.set_facecolor(BG)

bottom = np.zeros(len(footprint))
x_pos = np.arange(len(footprint))
for model in models:
    ax.bar(x_pos, footprint[model], bottom=bottom, label=model, color=model_colors[model], edgecolor=BG)
    bottom += footprint[model].values

ax.set_ylabel("Total energy (Joules, baseline + optimised runs)", color=FG)
ax.set_xlabel("Dataset", color=FG)
ax.set_title("Total energy footprint", color=FG, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels([d.replace("_", " ") for d in footprint.index], rotation=15, ha="right", color=FG)
ax.tick_params(axis="y", colors=FG)

legend = ax.legend(title="Model", fontsize=9, labelcolor=FG, frameon=False)
legend.get_title().set_color(FG)

for spine in ax.spines.values():
    spine.set_color(GRID)
sns.despine(ax=ax)

save(fig, "energy_footprint")
