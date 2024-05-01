import datetime
import random
import math

import csv


types = ["Charrue", "Semoir", "Tondeuse agricole",
         "Remorque agricole", "Tracteur", "Moissonneuse"]

etats = ["bon état", "besoin de réparation", "besoin de maintenance"]


def get_derniere_maintenance(etat, nb_panne, age):
    if etat == "besoin de maintenance":
        return datetime.datetime.now() - datetime.timedelta(days=random.randint(30, 60))
    elif etat == "besoin de réparation" and nb_panne > 5 and age > 6:
        return datetime.datetime.now() - datetime.timedelta(days=random.randint(60, 90))
    elif etat == "besoin de réparation":
        return datetime.datetime.now() - datetime.timedelta(days=random.randint(15, 30))
    else:
        return datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 15))


def determiner_classe(equipement):
    age_ratio = equipement["age"] / equipement["duree_vie_prevue"]
    etat = equipement["etat"]
    pannes = equipement["nb_pannes"]

    if etat == "bon état" and pannes < 10 and age_ratio < 0.5:
        return "A"
    elif (etat in ["bon état", "besoin de maintenance"] and pannes < 30 and 0.5 <= age_ratio < 0.75):
        return "B"
    elif (etat in ["bon état", "besoin de maintenance", "besoin de réparation"] and pannes < 50 and 0.75 <= age_ratio < 1):
        return "C"
    elif age_ratio >= 1 or pannes >= 50:
        return "C"
    else:
        return "B"


def creer_equipement():
    etat = random.choice(etats)
    age = random.randint(3, 13)
    nb_panne = math.ceil(random.randint(5, 10) * (age / 2.5))
    data = {
        "type_equipement": random.choice(types),
        "age": age,  # en années
        "duree_vie_prevue": random.randint(10, 16),  # en années
        "etat": etat,
        "derniere_maintenance": get_derniere_maintenance(etat, nb_panne, age),
        "frequence_maintenance": random.randint(1, 90),  # en jours
        "nb_pannes": nb_panne,
        "temps_fonctionnement_total": age * (24 - 10) * (365 - 100),
        "temps_moyen_reparation": random.randint(2, 18),  # en heures
        "temps_arret_panne": random.randint(2, 48)  # en heures,
    }

    data["class_equipement"] = determiner_classe(data)
    return data


n = 1000
if __name__ == "__main__":
    data = [creer_equipement() for _ in range(n)]
    with open("mock_data.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        for d in data:
            writer.writerow(d)
