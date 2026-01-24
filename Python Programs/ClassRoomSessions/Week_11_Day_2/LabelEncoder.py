from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(["Green","Yellow", "Orange", "Red", "red", "GREEN","Blue","BLUE"])

print("Classes learned by fit method:")
for idx, class_label in enumerate(le.classes_):
    print(f"  {idx} -> {class_label}")

encoded_labels = le.transform(["red", "Orange", "Red"])

print("Encoded labels:", encoded_labels)

print("\nEncoded value for each label:")
for label, encoded_val in zip(["red", "Orange", "Red"], encoded_labels):
    print(f"  {label} -> {encoded_val}")

original_labels = le.inverse_transform([5,1,2])

print("\nOriginal labels:", original_labels)

print("\nRetrieving each original label from encoded:")
for encoded_val, original_label in zip([5,1,2], original_labels):
    print(f"  {encoded_val} -> {original_label}")