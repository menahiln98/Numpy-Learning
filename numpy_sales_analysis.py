import numpy as np

# # The first task is to setup the data

sales = np.array([
    [15000, 18000, 22000, 19000, 25000, 21000],  # Hamza
    [12000, 14000, 13000, 16000, 15000, 17000],  # Sidra
    [20000, 22000, 25000, 28000, 30000, 27000],  # Waqas
    [ 8000,  9000,  7000, 11000, 10000, 12000],  # Faryal
    [18000, 20000, 19000, 22000, 24000, 23000],  # Hala
])

people = ["Hamza", "Sidra", "Waqas", "Faryal", "Hala"]
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
print("=== COMPANY OVERVIEW ===")
print("Shape of data:", sales.shape)
print("Total entries:", sales.size)

# # Basic Overview:

# Total sales of the entire company in 6 months:
print("Company sales:", np.sum(sales))

# Average sale per month per person:
print("Average per person:", np.mean(sales))

# Best single month sale ever:
print("Best sales were in:", np.max(sales))

# Worst single month sale ever:
print("Lowest sales were in:", np.min(sales))

# # Performance Per Salesperson

print("\n=== PERFORMANCE ===")
# Total earned by each person:
total = np.sum(sales, axis=1)
for i in range(len(people)):
    print(f"{people[i]} total: {total[i]}")

# Average per person:
averages = np.mean(sales, axis=1)
print("Averages:\n", averages)

# Best & Worst performer:
best_idx = np.argmax(total)
worst_idx = np.argmin(total)
print("Best performer is:\n", {people[best_idx]})
print("Worst performer is:\n", {people[worst_idx]})

# # Monthly Trends:

# How did company do each month?:
monthly_total = np.sum(sales, axis=0)
monthly_avg = np.mean(sales, axis=0)
print("Monthly records:\n")
for i in range(len(months)):
    print(f"{months[i]}: {monthly_total[i]}")

# Best & Worst month:
best_month = np.argmax(monthly_total)
worst_month = np.argmin(monthly_total)
print(f"Best month is:", {months[best_month]})
print(f"Worst month is:", {months[worst_month]})

# # Consistency Check:

print("\n=== HIGHLIGHTS ===")
# Who is most consistent?
consistency = np.std(sales, axis=1)
print("Consistency (lower = more consistent):")
for i in range(len(people)):
    print(f"  {people[i]}: {consistency[i]:.2f}")
most_consistent = np.argmin(consistency)
print(f"\nMost consistent: {people[most_consistent]}")

# # Filtering:

print("\n=== LABELS ===")
# Which months did Hamza cross 20000?:
Hamza = sales[0]
good_months = np.array(months)[Hamza > 20000]
print("Hamza crossed 20k in: ", good_months)

# Which people averaged above 18000 per month?
averages = np.mean(sales, axis=1)
high_performers = np.array(people)[averages > 18000]
print("High performers are:", high_performers)

# Flag each person: High / Low performer
labels = np.where(averages > 18000, "High", "Low")
for i in range(len(people)):
    print(f"  {people[i]}: {labels[i]}")

# # Normalize the Data:

# Scale all sales between 0 and 1
min_val = np.min(sales)
max_val = np.max(sales)
normalized = (sales - min_val) / (max_val - min_val)
print("Normalized sales (0 to 1):")
print(np.round(normalized, 2))

# # Correlation:

# Are any two people's sales patterns similar?
# Check Hamza vs Hala
corr = np.corrcoef(sales[0], sales[4])
print(f"\nHamza vs Hala correlation: {corr[0,1]:.2f}")

# Check all correlations at once
full_corr = np.corrcoef(sales)
print("\nFull correlation matrix:")
print(np.round(full_corr, 2))



