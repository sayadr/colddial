import pandas as pd

def map_source_to_destination(source_df: pd.DataFrame) -> pd.DataFrame:
    """
    Statically maps only known columns from source_df to destination fields.
    Excludes all 'Contact 1 *' and 'Contact 2 *' destination columns.
    """
    dest = pd.DataFrame()

    # --- Basic Property and Owner Info ---
    dest["Address"] = source_df.get("Address")
    dest["City"] = source_df.get("City")
    dest["State"] = source_df.get("State")
    dest["Zip Code"] = source_df.get("Zip Code")
    dest["Email"] = source_df.get("Email")

    dest["Owner Address"] = source_df.get("RecipientAddress")
    dest["Owner City"] = source_df.get("RecipientCity")
    dest["Owner State"] = source_df.get("RecipientState")
    dest["Owner Zip"] = source_df.get("RecipientPostalCode")
    dest["Estimated Owner Address"] = source_df.get("PropertyAddress")
    dest["Owner Occupied"] = source_df["AbsenteeOwner"].apply(
        lambda x: "Absentee" if x == "1" else "Owner Occupied"
    )
    dest["County"] = source_df.get("County")
    dest["Year Built"] = source_df.get("YearBuilt")
    dest["Square Footage"] = source_df.get("SquareFootage")
    dest["Lot Size"] = source_df.get("LotSizeSqFt")
    dest["Property Type"] = source_df.get("PropertyType")
    dest["Bathrooms"] = source_df.get("Baths")
    dest["Bedrooms"] = source_df.get("Beds")

    # --- MLS Info ---
    dest["MLS_Curr_ListAgentName"] = source_df.get("MLS_Curr_ListAgentName")
    dest["MLS_Curr_ListAgentPhone"] = source_df.get("MLS_Curr_ListAgentPhone")
    dest["MLS_Curr_ListAgentEmail"] = source_df.get("MLS_Curr_ListAgentEmail")
    dest["MLS_Curr_ListPrice"] = source_df.get("MLS_Curr_ListPrice")
    dest["MLS_Curr_Status"] = source_df.get("MLS_Curr_Status")
    dest["MLS Number"] = source_df.get("MLS_Curr_ListingID")

    # --- Pricing / Financial Fields ---
    dest["WholesaleValue"] = source_df.get("WholesaleValue")
    dest["LTV"] = source_df.get("LTV")
    dest["Original Purchase Price (A/B)"] = source_df.get("LastSalesPrice")
    dest["MarketValue"] = source_df.get("MarketValue")
    dest["AssessedTotal"] = source_df.get("AssessedTotal")

    # --- Phone and Seller Info (Primary) ---
    # Corrected: pull from source columns "Phone #1" and "Phone #1 Type"
    dest["Phone"] = source_df.get("Phone #1")
    dest["Phone Type"] = source_df.get("Phone #1 Type")
    # Combine first+last for Seller's Name (keeps destination name intact)
    dest["First Name"] = source_df.get("FirstName")
    dest["Last Name"] = source_df.get("LastName")

    # --- Misc and Derived ---
    dest["Status"] = source_df.get("MLS_Curr_Status")

    return dest