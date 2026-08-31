from research.v4 import factor_recipes


def test_default_factor_recipes_are_versioned_and_active(monkeypatch, tmp_path):
    monkeypatch.setattr(factor_recipes, "STORE", tmp_path / "recipes.json")
    recipes = factor_recipes.active_recipes()
    assert set(recipes) == {"market_regime", "momentum", "money_flow", "fundamental", "technical"}
    assert all(item["recipe_id"] and item["recipe_hash"] for item in recipes.values())


def test_recipe_activation_keeps_the_prior_recipe_in_history(monkeypatch, tmp_path):
    monkeypatch.setattr(factor_recipes, "STORE", tmp_path / "recipes.json")
    current = factor_recipes.active_recipes()["momentum"]
    replacement = factor_recipes.create_recipe("momentum", current["components"], current["parameters"], "exp_source")
    changed = factor_recipes.set_active_recipe("momentum", replacement["recipe_id"], "exp_source", "manual review")
    assert changed["previous_recipe_id"] == current["recipe_id"]
    assert factor_recipes.active_recipes()["momentum"]["recipe_id"] == replacement["recipe_id"]
