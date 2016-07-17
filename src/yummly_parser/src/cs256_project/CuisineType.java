package cs256_project;

public enum CuisineType {

	brazilian,
	british,
	cajun_creole,
	chinese,
	filipino,
	french,
	greek,
	indian,
	irish,
	italian,
	jamaican,
	japanese,
	korean,
	mexican,
	moroccan,
	russian,
	southern_us,
	spanish,
	thai,
	vietnamese;
	
	private static int totalCuisineTypes;
	
	CuisineType(){
//		recipeCount = 0;
	}
	
//	/**
//	 * Returns the cuisine type's ID.  This is useful when
//	 * storing cuisine type information in an array.
//	 * 
//	 * @return Identification number between 0 and the Number of Cuisine Types (e.g. 20) minus 1.
//	 */
//	public int getID(){
//		return id;
//	}
//	
//	
//	/**
//	 * Returns the name of the enumerated cuisine type as a String.
//	 * 
//	 * @return  String - enumerated object's name
//	 */
//	public String getName(){
//		return name;
//	}
	
//	/**
//	 * Each enumerated object contains a count for the number of recipes of that
//	 * cuisine type.  This tracks that counter.
//	 */
//	public void incrementRecipeCount(){
//		recipeCount++;
//	}
//	
//	/**
//	 * Returns the number of recipes in the dataset for this cuisine type.
//	 * 
//	 * @return Integer - Number of recipes in the dataset of this cuisine type.
//	 */
//	public int getRecipeCount(){
//		return recipeCount;
//	}
	
	/**
	 * 
	 * Returns the Cuisine Type number corresponding to the 
	 * 
	 * @param ordinalNumb - CuisineType ordinal number.
	 * 
	 * @return CuisineType that corresponds to the specified ordinal number.
	 */
	public static CuisineType fromInt(int ordinalNumb){
		return CuisineType.values()[ordinalNumb];
	}
	
	
	@Override
	public String toString(){
		return this.name();
	}
	
	
	/**
	 * 
	 * Extracts the number of different supported cuisine types.
	 * 
	 * @return Number of different cuisine types
	 */
	public static int count(){
		// Determine the number of cuisine types if it is not already stored.
		if(totalCuisineTypes == 0 ) 
			totalCuisineTypes = CuisineType.values().length;
		return totalCuisineTypes;
	}
	
}
