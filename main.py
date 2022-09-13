from src.test import NBAevaluator
import logging

if __name__ == "__main__":
    logging.info("main")
    evaluator = NBAevaluator()
    test_records = evaluator.fitting_pipeline(gs = True)
    evaluator.select_save_best_model(model_name = 'logreg')

