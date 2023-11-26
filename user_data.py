class UserData:
    def __init__(self, user_id, expenses, credit_score=None):
        self.user_id = user_id
        self.expenses = expenses
        self.credit_score = credit_score

    def get_user_id(self):
        return self.user_id

    def get_expenses(self):
        return self.expenses

    def get_credit_score(self):
        return self.credit_score

    def set_credit_score(self, new_credit_score):
        self.credit_score = new_credit_score
