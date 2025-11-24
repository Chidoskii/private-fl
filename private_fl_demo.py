# Starter code for Homework
import numpy as np
import pandas as pd

# Problem setup

# Update to point to the dataset on your machine
data: pd.DataFrame = pd.read_csv(
    "https://raw.githubusercontent.com/opendp/cs208/refs/heads/main/spring2025/data/fake_healthcare_dataset_sample100.csv"
)

# names of public identifier columns
pub = ["age", "sex", "blood", "admission"]

# variable to reconstruct
target = "result"


def execute_subsetsums_exact(predicates):
    """Count the number of patients that satisfy each predicate.
    Resembles a public query interface on a sequestered dataset.
    Computed as in equation (1).

    :param predicates: a list of predicates on the public variables
    :returns a 1-d np.ndarray of exact answers the subset sum queries"""
    return data[target].values @ np.stack([pred(data) for pred in predicates], axis=1)


def execute_subsetsums_round(multiple, predicates):
    exact_count = data[target].values @ np.stack(
        [pred(data) for pred in predicates], axis=1
    )
    rounded_count = (np.round(exact_count / multiple) * multiple).astype(int)
    return rounded_count


def execute_subsetsums_noise(sigma, predicates):
    exact_count = data[target].values @ np.stack(
        [pred(data) for pred in predicates], axis=1
    )
    guassian_sums = exact_count + np.random.normal(
        loc=0, scale=sigma, size=len(predicates)
    ).astype(int)
    return guassian_sums


def execute_subsetsums_sample(sample_size, predicates):
    data_length = len(data)
    sample_indices = np.random.choice(data_length, size=sample_size, replace=False)
    sample_data = data.iloc[sample_indices]
    exact_sample_count = sample_data[target].values @ np.stack(
        [pred(sample_data) for pred in predicates], axis=1
    )
    scaled_sums = (exact_sample_count * (data_length / sample_size)).astype(int)
    return scaled_sums


def make_random_predicate():
    """Returns a (pseudo)random predicate function by hashing public identifiers."""
    prime = 2003
    desc = np.random.randint(prime, size=len(pub))
    # this predicate maps data into a 1-d ndarray of booleans
    #   (where `@` is the dot product and `%` modulus)
    return lambda data: ((data[pub].values @ desc) % prime % 2).astype(bool)


def find_rmse(true_answers, approx_answers):
    errors = true_answers - approx_answers
    rmse = np.sqrt(np.mean(errors**2))
    return rmse


def attack_success(values, constructed):
    if len(values) != len(constructed):
        raise ValueError("Lists must be of equal length")
    correct_bits = sum(1 for v, c in zip(values, constructed) if v == c)
    success_rate = correct_bits / len(values)
    return success_rate


def reconstruction_attack(data_pub, predicates, answers):
    """Reconstructs a target column based on the `answers` to queries about `data`.

    :param data_pub: data of length n consisting of public identifiers
    :param predicates: a list of k predicate functions
    :param answers: a list of k answers to a query on data filtered by the k predicates
    :return 1-dimensional boolean ndarray"""
    predicate_matrix = np.stack(
        [pred(data_pub).astype(int) for pred in predicates], axis=1
    )
    x_hat, *_ = np.linalg.lstsq(predicate_matrix.T, answers, rcond=None)
    x_hat = (x_hat >= 0.5).astype(int)
    return x_hat


if __name__ == "__main__":
    # EXAMPLE: writing and using predicates
    num_female_patients, num_emergency_admits = execute_subsetsums_exact(
        [
            lambda data: data["sex"] == 1,  # "is-female" predicate
            lambda data: data["admission"] == 2,  # "had emergency admission" predicate
        ]
    )

    # print(num_female_patients)
    # print(num_emergency_admits)
    # EXAMPLE: making and using a random predicate
    # example_predicate = make_random_predicate()
    # num_patients_that_matched_random_predicate = execute_subsetsums_exact(
    #     [example_predicate]
    # )
    # print(num_patients_that_matched_random_predicate)

    # The boolean mask from applying the example predicate to the data:
    # example_predicate_mask = example_predicate(data)

    noisy_round_value = 99
    round_value = noisy_round_value
    noise_value = noisy_round_value
    subset_value = 100
    predicates = []
    num_of_queries = 2 * len(data)
    i = 0
    while i < num_of_queries:
        new_predicate = make_random_predicate()
        predicates.append(new_predicate)
        i += 1

    answers = execute_subsetsums_exact(predicates)
    rounded_answers = execute_subsetsums_round(round_value, predicates)
    noisy_answers = execute_subsetsums_noise(noise_value, predicates)
    subset_answers = execute_subsetsums_sample(subset_value, predicates)

    sensitive_bits = reconstruction_attack(data, predicates, answers)
    rounded_sensitive_bits = reconstruction_attack(data, predicates, rounded_answers)
    noisy_sensitive_bits = reconstruction_attack(data, predicates, noisy_answers)
    subset_sensitive_bits = reconstruction_attack(data, predicates, subset_answers)

    rounded_rsme = find_rmse(answers, rounded_answers)
    noisy_rsme = find_rmse(answers, noisy_answers)
    subset_rsme = find_rmse(answers, subset_answers)

    rounded_success = attack_success(sensitive_bits, rounded_sensitive_bits)
    noisy_success = attack_success(sensitive_bits, noisy_sensitive_bits)
    subset_success = attack_success(sensitive_bits, subset_sensitive_bits)

    rounded_results = "Rounded Results \nMultiple: {2} \t Accuracy: {0} \t Success: {1}"
    noisy_results = "Noisy Results \nNoise Level: {2} \t Accuracy: {0} \t Success: {1}"
    subset_results = (
        "Subset Results \nSubset Size: {2} \t Accuracy: {0} \t Success: {1}"
    )

    print(rounded_results.format(rounded_rsme, rounded_success, round_value))
    print(noisy_results.format(noisy_rsme, noisy_success, noise_value))
    print(subset_results.format(subset_rsme, subset_success, subset_value))
