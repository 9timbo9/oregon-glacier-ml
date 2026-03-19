import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Input data
# ----------------------------
data = {
    "year": [1986, 1986, 1986, 1986,
             2000, 2000, 2000, 2000,
             2020, 2020, 2020, 2020,
             2025, 2025, 2025, 2025],
    "patch": ["patch_001", "patch_002", "patch_003", "patch_004"] * 4,
    "area_km2": [
        16.494363490123828, 12.23513113416435, 21.181260562456306, 16.05057527205591,
        17.68728368323163, 11.372914779322535, 20.410208250760896, 13.43121999867844,
        10.604942764545292, 4.450704611901652, 5.700344384345843, 2.887635259854883,
        6.086882078727739, 2.748937947297985, 4.860350924781194, 2.6837062160798206
    ]
}

df = pd.DataFrame(data)

# Make output cleaner
df["patch_label"] = df["patch"].str.replace("patch_", "Patch ")

# ----------------------------
# 1. Total area by year
# ----------------------------
total_by_year = df.groupby("year", as_index=False)["area_km2"].sum()

plt.figure(figsize=(8, 5))
plt.bar(total_by_year["year"].astype(str), total_by_year["area_km2"])
plt.title("Total Mapped Snow and Ice Area by Year")
plt.xlabel("Year")
plt.ylabel("Total Area (km²)")
for i, v in enumerate(total_by_year["area_km2"]):
    plt.text(i, v + 0.8, f"{v:.2f}", ha="center")
plt.tight_layout()
plt.savefig("total_area_by_year.png", dpi=300)
plt.show()

# ----------------------------
# 2. Patch area by year (grouped bar chart)
# ----------------------------
pivot = df.pivot(index="year", columns="patch_label", values="area_km2")

x = np.arange(len(pivot.index))
width = 0.2

plt.figure(figsize=(10, 6))
for i, col in enumerate(pivot.columns):
    plt.bar(x + i * width - 1.5 * width, pivot[col], width=width, label=col)

plt.xticks(x, pivot.index.astype(str))
plt.xlabel("Year")
plt.ylabel("Area (km²)")
plt.title("Mapped Snow and Ice Area by Patch and Year")
plt.legend()
plt.tight_layout()
plt.savefig("patch_area_grouped_bar.png", dpi=300)
plt.show()

# ----------------------------
# 3. Line chart for each patch over time
# ----------------------------
plt.figure(figsize=(9, 6))
for patch in df["patch_label"].unique():
    sub = df[df["patch_label"] == patch]
    plt.plot(sub["year"], sub["area_km2"], marker="o", label=patch)

plt.title("Snow and Ice Area Change Over Time by Patch")
plt.xlabel("Year")
plt.ylabel("Area (km²)")
plt.xticks([1986, 2000, 2020, 2025])
plt.legend()
plt.tight_layout()
plt.savefig("patch_area_line_chart.png", dpi=300)
plt.show()

# ----------------------------
# 4. Percent change from 1986 to 2025 by patch
# ----------------------------
area_1986 = df[df["year"] == 1986][["patch_label", "area_km2"]].rename(columns={"area_km2": "area_1986"})
area_2025 = df[df["year"] == 2025][["patch_label", "area_km2"]].rename(columns={"area_km2": "area_2025"})

change_df = area_1986.merge(area_2025, on="patch_label")
change_df["percent_change"] = ((change_df["area_2025"] - change_df["area_1986"]) / change_df["area_1986"]) * 100

plt.figure(figsize=(8, 5))
plt.bar(change_df["patch_label"], change_df["percent_change"])
plt.axhline(0, linewidth=1)
plt.title("Percent Change in Snow and Ice Area by Patch (1986–2025)")
plt.xlabel("Patch")
plt.ylabel("Percent Change (%)")
for i, v in enumerate(change_df["percent_change"]):
    plt.text(i, v - 3 if v < 0 else v + 1, f"{v:.1f}%", ha="center")
plt.tight_layout()
plt.savefig("percent_change_by_patch.png", dpi=300)
plt.show()

# ----------------------------
# 5. Print useful summary stats
# ----------------------------
print("\nTotal area by year:")
print(total_by_year)

start_total = total_by_year.loc[total_by_year["year"] == 1986, "area_km2"].values[0]
mid_total = total_by_year.loc[total_by_year["year"] == 2000, "area_km2"].values[0]
end20_total = total_by_year.loc[total_by_year["year"] == 2020, "area_km2"].values[0]
end25_total = total_by_year.loc[total_by_year["year"] == 2025, "area_km2"].values[0]

print("\nPercent changes:")
print(f"1986 -> 2000: {((mid_total - start_total) / start_total) * 100:.1f}%")
print(f"2000 -> 2020: {((end20_total - mid_total) / mid_total) * 100:.1f}%")
print(f"2020 -> 2025: {((end25_total - end20_total) / end20_total) * 100:.1f}%")
print(f"1986 -> 2025: {((end25_total - start_total) / start_total) * 100:.1f}%")