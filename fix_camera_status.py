import pandas as pd

# Read the CSV file
df = pd.read_csv("data/cctv_config.csv")
print("Current CCTV Configuration:")
print("=" * 50)
print(df.to_string(index=False))

print("\nStatus Distribution:")
print(df["status"].value_counts())

# Test updating one camera status
print("\nUpdating disconnected cameras to 'disconnect' status...")

# Find cameras that are likely disconnected and update them
df.loc[df["id"].isin(["1001", "1002", "c3093399-6335-4ab2-b6ec-c606fd0b063d"]), "status"] = "disconnect"

# Save the updated file
df.to_csv("data/cctv_config.csv", index=False)

print("\nUpdated CCTV Configuration:")
print("=" * 50)
print(df.to_string(index=False))

print("\nUpdated Status Distribution:")
print(df["status"].value_counts())
