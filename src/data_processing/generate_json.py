import json
import os
import sys
from pathlib import Path
import pandas as pd
from config.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.logger import setup_logger


def generate_places_reviews_json(
    places_path: str,
    reviews_path: str,
    output_path: str = "places_reviews.json"
) -> None:
    """
    Merges place and review data into a single JSON file where each place contains its associated reviews.

    Parameters
    ----------
    places_path : str
        Path to the CSV file containing place information. Expected columns:
        ['Name', 'Street', 'Neighborhood', 'City', 'Rating', 'Number of Reviews',
         'Place ID', 'Type', 'Latitude', 'Longitude'].

    reviews_path : str
        Path to the CSV file containing review information. Expected columns:
        ['Place ID', 'Place Name', 'Review ID', 'Author', 'Rating', 'Text',
         'Review Length', 'Word Count', 'Time', 'Date', 'Response'].

    output_path : str, optional
        Path to save the output JSON file. Defaults to 'places_reviews.json'.

    Returns
    -------
    None
        The function writes the combined output to a JSON file.
    
    Notes
    -----
    - This function assumes both CSV files use ';' as a separator.
    - Reviews are grouped by Place ID and embedded into their corresponding place entries.
    """

    # Read both CSV files
    logger.info(f"Loading places from: {places_path}")
    places_df = pd.read_csv(places_path, sep=';')
    logger.info(f"Loaded {len(places_df)} places")

    logger.info(f"Loading reviews from: {reviews_path}")
    reviews_df = pd.read_csv(reviews_path, sep=';')
    logger.info(f"Loaded {len(reviews_df)} reviews")

    # Group reviews by Place ID for faster lookup
    grouped_reviews = reviews_df.groupby("Place ID")

    # Final output list
    result = []

    # Iterate through each place
    for idx, place in places_df.iterrows():
        place_id = place["Place ID"]

        # Extract all reviews associated with this place
        reviews = []
        if place_id in grouped_reviews.groups:
            place_reviews = grouped_reviews.get_group(place_id)
            for _, review in place_reviews.iterrows():
                reviews.append({
                    "review_id": review.get("Review ID"),
                    "author": review.get("Author"),
                    "rating": review.get("Rating"),
                    "text": review.get("Text"),
                    "review_length": review.get("Review Length"),
                    "word_count": review.get("Word Count"),
                    "time": review.get("Time"),
                    "date": review.get("Date"),
                    "response": review.get("Response")
                })
        else:
            logger.warning(f"No reviews found for place_id: {place_id} ({place.get('Name')})")                

        # Create JSON object for the place
        result.append({
            "place_id": place_id,
            "name": place.get("Name"),
            "street": place.get("Street"),
            "neighborhood": place.get("Neighborhood"),
            "city": place.get("City"),
            "rating": place.get("Rating"),
            "num_reviews": place.get("Number of Reviews"),
            "type": place.get("Type"),
            "latitude": place.get("Latitude"),
            "longitude": place.get("Longitude"),
            "reviews": reviews
        })

        if idx % 100 == 0 and idx > 0:
            logger.info(f"Processed {idx} places")        

    # Write result to JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved merged JSON to: {Path(output_path).resolve()}")

    print(f"JSON file saved to: {Path(output_path).resolve()}")


if __name__ == "__main__":
    # Execute the data merging function when running this script directly
    
    logger = setup_logger(name="generate_json", log_filename="generate_json.log")
    logger.info("Starting logger process.")
    
    try:
        generate_places_reviews_json(
            places_path=os.path.join(RAW_DATA_DIR, "places.csv"),         # Path to the raw places CSV file
            reviews_path=os.path.join(RAW_DATA_DIR, "reviews.csv"),       # Path to the raw reviews CSV file
            output_path=os.path.join(PROCESSED_DATA_DIR, "places_reviews.json")  # Output path for the merged JSON file
        )
        logger.info("Finished generate_places_reviews_json successfully.")
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)