qf7 = pd.read_csv("International Astronaut Database.csv")
#i made the df "qf7" to not interfere with other people's dfs.
def earliest_year(s):
    if pd.isna(s): return np.nan
    yrs = re.findall(r"\((\d{4})\)", str(s))
    return int(min(yrs)) if yrs else np.nan

qf7["first_year"] = qf7["Flights"].apply(earliest_year)

#4 group mapping related to the question about ussr, russia, and USA.
core = {"United States", "Soviet Union", "Russia"}
def to_group(c):
    return c if c in core else "Others"

qf7["Country4"] = qf7["Country"].apply(to_group)

yearly = (qf7.dropna(subset=["first_year"])
             .groupby(["first_year","Country4"])
             .size().rename("count").reset_index())
wide = yearly.pivot(index="first_year", columns="Country4", values="count").fillna(0)

years_full = range(int(wide.index.min()), int(wide.index.max())+1)
wide = wide.reindex(years_full, fill_value=0)

plt.figure(figsize=(11,6))
for col in ["United States", "Soviet Union", "Russia", "Others"]:
    if col in wide.columns:
        plt.plot(wide.index, wide[col], label=col)

plt.xlabel("Year (first spaceflight)")
plt.ylabel("Astronauts")
plt.title("Astronauts per Year by Country Group")
plt.legend()
plt.tight_layout()
plt.show()