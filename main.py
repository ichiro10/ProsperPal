# main.py
from budget_management import BudgetManagement

def main():
    # Exemple d'utilisation
    user_data = {"user_id": 123, "expenses": [100, 150, 200, 50]}
    
    # Initialisation du gestionnaire de budget
    budget_manager = BudgetManagement(user_data)

    

    # Analyse des habitudes de dépenses
    budget_manager.analyze_spending_habits()

    # Génération du plan budgétaire
    budget_manager.generate_budget_plan()

    # Notification de l'utilisateur
    budget_manager.notify_user("Your budget plan has been generated!")

if __name__ == "__main__":
    main()