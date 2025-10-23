import random
import pandas as pd

def generate_crop_data(start_year=1990, end_year=2030):
    """
    Generates simulated historical and future crop data.

    Args:
        start_year (int): The starting year for data generation.
        end_year (int): The ending year for data generation.

    Returns:
        list: A list of dictionaries, each representing crop data for a year.
    """
    crop_types = ['Wheat', 'Rice', 'Corn', 'Soybeans', 'Barley']
    weather_conditions = ['Favorable', 'Normal', 'Drought', 'Heavy Rain', 'Early Frost']
    yield_accuracies = [f"{random.randint(70, 95)}%" for _ in range(len(crop_types))]

    data = []
    for year in range(start_year, end_year + 1):
        crop_type = random.choice(crop_types)
        weather = random.choice(weather_conditions)
        yield_acc = random.choice(yield_accuracies)

        # Simulate some trends or variations
        if 'Drought' in weather:
            yield_acc = f"{random.randint(40, 60)}%"
        elif 'Favorable' in weather:
            yield_acc = f"{random.randint(90, 98)}%"

        data.append({
            'year': year,
            'crop_type': crop_type,
            'yield_accuracy': yield_acc,
            'weather_conditions': weather
        })
    return data

if __name__ == '__main__':
    historical_data = generate_crop_data(start_year=1990, end_year=2025)
    future_data = generate_crop_data(start_year=2026, end_year=2030)

    print("Historical Crop Data (1990-2025):")
    for entry in historical_data:
        print(entry)

    print("\nFuture Crop Predictions (2026-2030):")
    for entry in future_data:
        print(entry)
