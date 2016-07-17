package cs256_project;

import java.util.Comparator;

public class Ingredient {

	private String name;
	private int[] cuisineUsage;
	private int totalRecipeCount = 0;
	private double entropy = -1;
	
	
	public Ingredient(String name){
		this.name = name;
		
		// Initialize the array of cuisine type counts.
		CuisineType[] allTypes = CuisineType.values();
		cuisineUsage = new int[allTypes.length];
		
		// Initialize the array of usage counts for all cuisine types to 0.
		for(int i = 0; i < cuisineUsage.length; i++)
			cuisineUsage[i] =0;
		
		entropy = -1;
	}
	
	
	/**
	 * Returns the name of the ingredient as a string.
	 * 
	 * @return String - Name of the ingredient
	 */
	public String getName(){
		return name;
	}
	
	/**
	 * Increments the usage of specific cuisine type for this recipe.
	 * 
	 * @param type Cuisine Type whose count will be incremented.
	 */
	public void incrementCuisineTypeCount(CuisineType type){
		int ordinal = type.ordinal();
		cuisineUsage[ordinal]++;
		
		// Increment the total number of times this ingredient is used.
		totalRecipeCount++;
	}
	
	
	
	/**
	 * Extracts the number of times this ingredient is used for each
	 * cuisine type.
	 * 
	 * @return  int[] - For each of the CuisineType, this function returns the 
	 * number of recipes of that type containing this ingredient.
	 */
	public int[] getAllCuisineTypesCount(){
		int[] typeUsage = new int[cuisineUsage.length];
		
		for(int i = 0; i < typeUsage.length; i++)
			typeUsage[i] = cuisineUsage[i];
		
		return typeUsage;
	}
	
	
	/**
	 * Extracts the number of types this ingredient is used
	 * in the specified cuisine type.
	 * 
	 * @param id - CuisineType ID
	 * @return Number of recipes of the specified cuisine type ID this recipe appears	
	 */
	public int getCuisineTypeCount(int id){ return cuisineUsage[id];}
	
	
	/**
	 * Calculates the entropy of a value using the different
	 * cuisine types as class values.
	 * 
	 * entropy = - SUM_(i=1)^k ( P(v_i) * log2(P(v_i))
	 * 
	 * * k - Number of class values
	 * 
	 * * P(v_i) - Probability of the i-th class value.
	 * 
	 * For more information, see page of "Introduction to Data Mining"
	 * by Tan et. al. 
	 * 
	 * @return double - Value of Ingredient's Entropy
	 */
	public double getEntropy(){
		double p_i = 0;
		
		// Only calculate entopy once.
		if( entropy >= 0) return entropy;
		
		entropy = 0; // Reset entropy
		for(int i = 0; i < cuisineUsage.length; i ++){
			if(cuisineUsage[i] != 0){
				p_i = (float) cuisineUsage[i] / totalRecipeCount;
				entropy -= p_i * (Math.log(p_i) / Math.log(2));
			}
		}
		return entropy;
	}
	
	
	/**
	 * Gets the recipe count for this ingredient.
	 * 
	 * @return Total number of recipes containing this ingredient.
	 */
	public int getTotalRecipeCount(){ return totalRecipeCount;	}
	
	/**
	 * 
	 * Class object that allows for the sorting of ingredients by name.
	 * 
	 * @author Zayd
	 *
	 */
	public static class NameCompare implements Comparator<Ingredient>{
	    @Override
	    public int compare(Ingredient i1, Ingredient i2) {
	        return i1.getName().compareTo(i1.getName());
	    }
	}
	
	
	/**
	 * 
	 * Class object that allows for the sorting of ingredients by name.
	 * 
	 * @author Zayd
	 *
	 */
	public static class RecipeCountCompareDescending implements Comparator<Ingredient>{
	    @Override
	    public int compare(Ingredient i1, Ingredient i2) {
	        return i2.getTotalRecipeCount() - i1.getTotalRecipeCount() ;
	    }
	}
	

}
