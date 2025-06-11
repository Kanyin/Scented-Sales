def to_rebrand(brand):
    return( not brand or 
            brand.lower() == 'unbranded' or
            brand.lower().startswith('as')
          )



def rebrand(row):
    global new_brand
    if to_rebrand(row['brand']):
        print(f"\nTitle: {row['title']}")
        print(f"Current brand: '{row['brand']}'")
        new_brand = input("Enter correct brand (or press Enter to keep existing): ")
        return new_brand if new_brand else row['brand']
    else:
        return row['brand']
