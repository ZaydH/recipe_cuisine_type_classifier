package cs256_project;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Hashtable;
import java.util.List;
import java.util.Scanner;
import java.util.Set;

import javax.swing.JOptionPane;


public class RecipeCollection {

	private Hashtable<Integer, Recipe> allRecipes = new Hashtable<Integer,Recipe>();
	private Hashtable<CuisineType, Integer> cuisineTypeCount = new Hashtable<CuisineType,Integer>();
	private Hashtable<String, Ingredient> allIngredients = new Hashtable<String,Ingredient>();
	private int badRecordCount = 0;
	ValueDifferenceMetricCompare vdmCompare = new ValueDifferenceMetricCompare(); 
	
	
	private static final int MINIMUM_RECIPE_SIZE = 4;
	
	private static final int INCREMENTAL_PRINT_FREQUENCY = 1000;
	
	public static void main(String[] args){
		
		if(args.length != 1){
			System.out.println("Error: Invalid number of input arguments.  A single input argument is required.");
			return;
		}
		
	    // Collect Data on different Settings
		try{
			String filePath = "algorithm_comparison.csv";
			BufferedWriter fileOut = new BufferedWriter(new FileWriter(filePath)); // Open the file containing the algorithm comparison results.

			fileOut.write("Algorithm_Name,k,FirstChoiceAccuracy,TopTwoAccuracy");
			fileOut.newLine();
			
			for(int k = 1; k <=64; k*=2)
				for(int randomSubsamplingCount = 0; randomSubsamplingCount < 33; randomSubsamplingCount++){
					
					//RecipeCollection fullCollection = RecipeCollection.getRecipeCollection("filtered_train.json.txt");
					RecipeCollection fullCollection = RecipeCollection.getRecipeCollection(args[0]);
					fullCollection.print("filtered_train.json.txt");
					
					// Define the training and test sets
					RecipeCollection[] cols = fullCollection.performRecipeHoldoutSplit((float)2/3);
					RecipeCollection trainingSet = cols[0];
					RecipeCollection testSet = cols[1];
					
					/*// Perform K-Nearest neighbor using Weighted Overlap
					RecipeCollection.OverlapCoefficientAmalgamated  amalOverlap = new RecipeCollection.OverlapCoefficientAmalgamated();
					RecipeResult knnAmalOverlap = trainingSet.performKNearestNeighbor(testSet, k, amalOverlap, false, false, true);
					fileOut.write("AmalOverlap," + k + "," + knnAmalOverlap.accuracy + "," + knnAmalOverlap.topTwoAccuracy);
					fileOut.newLine();*/
					
					// Perform K-Nearest neighbor using Weighted Overlap
					RecipeCollection.WeightedOverlapCoefficient weightedOverlapTemp = trainingSet.getWeightedOverlapCoefficient();
					RecipeResult knnResultWeightedOverlap = trainingSet.performKNearestNeighbor(testSet, 8, weightedOverlapTemp, false, false, true);
					fileOut.write("WeightedOverlapKnnResult," + k + "," + knnResultWeightedOverlap.accuracy + "," + knnResultWeightedOverlap.topTwoAccuracy);
					fileOut.newLine();
					
					RecipeCollection.AmalgamatedWeightedOverlapCoefficient amalWeightedOverlap = trainingSet.getAmalgamatedWeightedOverlapCoefficient();
					RecipeResult knnAmalWeightedOverlap = trainingSet.performKNearestNeighbor(testSet, k, amalWeightedOverlap, false, false, true);
					fileOut.write("AmalgamatedWeightedOverlap," + k + "," + knnAmalWeightedOverlap.accuracy + "," + knnAmalWeightedOverlap.topTwoAccuracy);
					fileOut.newLine();
					
					/*// Perform Naive Bayes
					RecipeResult naiveBayesResult = trainingSet.performNaiveBayes(testSet, new LaplaceIngredientClassProbability(), true);
					fileOut.write("NaiveBayes," + naiveBayesResult.accuracy + "," + naiveBayesResult.topTwoAccuracy);
					fileOut.newLine();
					
					// Perform K-Nearest neighbor using Modified Value Distance Metric
					RecipeCollection.ValueDifferenceMetricCompare mvdmCompare = trainingSet.getValueDifferenceMetricCompare();
					RecipeResult mvdmKnnResult = trainingSet.performKNearestNeighbor(testSet, 8, mvdmCompare, false, false, true);
					fileOut.write("MVDMKnnResult," + mvdmKnnResult.accuracy + "," + mvdmKnnResult.topTwoAccuracy);
					fileOut.newLine();
					
					// Perform K-Nearest neighbor using Weighted Overlap
					RecipeCollection.WeightedOverlapCoefficient weightedOverlapTemp = trainingSet.getWeightedOverlapCoefficient();
					RecipeResult knnResultWeightedOverlap = trainingSet.performKNearestNeighbor(testSet, 8, weightedOverlapTemp, false, false, true);
					fileOut.write("WeightedOverlapKnnResult," + knnResultWeightedOverlap.accuracy + "," + knnResultWeightedOverlap.topTwoAccuracy);
					fileOut.newLine();
					
					RecipeResult[] combinedResults = new RecipeResult[3];
					int cnt = 0;
					combinedResults[cnt++] = knnResultWeightedOverlap;
					combinedResults[cnt++] = mvdmKnnResult;
					combinedResults[cnt++] = naiveBayesResult;
					
					// Perform ensemble
					RecipeResult ensembleResult = testSet.performKNNandBayesEnsemble(combinedResults, true);
					fileOut.write("EnsembleResult," + ensembleResult.accuracy + "," + ensembleResult.topTwoAccuracy);
					fileOut.newLine();*/
					
	
	
				}
			// Put the end of the JSON file then close it.
			fileOut.close();

		}
		catch(IOException e){
			System.out.print("Error: Unable to write output file for the recipes.");
		}
		
		
		//System.out.println("Accuracy is: " + String.format("%9.2f", result.getAccuracy() * 100) + "%.");
	}

	
	/**
	 * Private Constructor for a Recipe Collection.
	 * 
	 * I do not want users creating Recipe Collections directly.  Instead,
	 * I will use a factory method object to create the recipe for them.
	 */
	private RecipeCollection(){
		
		// Initialize the cuisine list.
		CuisineType allTypes[] = CuisineType.values();
		for(CuisineType type : allTypes){
			this.cuisineTypeCount.put(type, 0);
		}
		
	}
	
	
	/**
	 * 
	 * Factory method to build a recipe collection from a file.
	 * 
	 * @param filePath		File containing the recipe information.
	 * @return				RecipeCollection if file parsing successful.  Null otherwise.
	 */
	public static RecipeCollection getRecipeCollection(String filePath){
		
		File file = new File(filePath);
		
		//----- Once the file extension has been verified, try opening the file.
		Scanner fileIn;
		try{
			fileIn = new Scanner(new FileReader(file));
		}
		catch(FileNotFoundException e){
			JOptionPane.showMessageDialog(null, "No file with the specified name exists.  Please specify a valid file and try again.");
			return null;
		}
	
		// Initialize the RecipeCollection to output
		RecipeCollection tempRC = new RecipeCollection();


		// Iterate through the recipe file and build the 
		final String RECORD_START_CHAR = "{";
		final String RECORD_END_CHAR = "}";
		String line;
		StringBuffer recipeInfo = new StringBuffer();
		while(fileIn.hasNextLine()){
			
			line = fileIn.nextLine();
			// Check if the StringBuffer needs to be cleared.
			if(line.indexOf(RECORD_START_CHAR) != -1 ) recipeInfo.setLength(0);
			
			// Append line to the String Buffer
			recipeInfo.append(line);
			recipeInfo.append("\n");
			
			// Find the end of the record
			if(line.indexOf(RECORD_END_CHAR) != -1 ){
				line = line.replace("},", "}");
				recipeInfo.append(line);
				Recipe newRecipe = Recipe.getRecipe(recipeInfo.toString());
				// Check for bad records.
				if(newRecipe == null){
					String badRecipeInfo = recipeInfo.toString();
					
					System.out.print("Bad Record: " + badRecipeInfo + "\n\n");
					tempRC.badRecordCount++; // Increment the bad record counter
					continue; // Return to the next while loop.
				}
				
				if(newRecipe.getIngredientCount() >= MINIMUM_RECIPE_SIZE)
					// Add new recipe to the list
					tempRC.addRecipe(newRecipe);
				else{
					if(tempRC.badRecordCount == 0) 
						System.out.println("Invalid recipes: (RecipeID,NumberOfIngredients)");
					System.out.println(newRecipe.getID() + "," + newRecipe.getIngredientCount());
					tempRC.badRecordCount++;
				}

			} //if(line.indexOf(RECORD_END_CHAR) > 0 )
		} //while(fileIn.hasNextLine())
		
		/*int[] wordCount = new int[25];
		int[] totalIngredientWordCount = new int[25];
        Set<String> keys = tempRC.allIngredients.keySet();
		for(String key : keys){
			int numbWords = key.split(" ").length;
			wordCount[numbWords]++;
			totalIngredientWordCount[numbWords] += tempRC.allIngredients.get(key).getTotalRecipeCount();
		}*/
		
		fileIn.close(); // Close the scanner
		return tempRC;	// Return the collection of recipe information.
	}
	
	/**
	 * Adds a new recipe to this collection
	 * 
	 * @param newRecipe Recipe to add to the collection
	 */
	private void addRecipe(Recipe newRecipe){ 
		allRecipes.put(newRecipe.getID(), newRecipe);
		
		// Update the cuisine type count in the hashtable
		CuisineType type = CuisineType.valueOf(newRecipe.getCuisineType());
		Integer cuisineCount = this.cuisineTypeCount.get(type);
		this.cuisineTypeCount.put(type, cuisineCount.intValue() + 1);

		// Update recipe frequency count
		String[] recipeIngredientNames = newRecipe.getIngredients();
		for(String ingredientName : recipeIngredientNames){
			
			// Extract the ingredient if it already exists.  
			Ingredient ingredient = this.allIngredients.get(ingredientName);
			// Build a new ingredient if this ingredient does not already exist.
			if(ingredient == null)	ingredient = new Ingredient(ingredientName);
			
			// Increment the usage of the ingredient for this recipe's cuisine type
			ingredient.incrementCuisineTypeCount(type);
			// Update the ingredients hash table
			this.allIngredients.put(ingredientName, ingredient);
		}
	
	}
	
	
	/**
	 * 
	 * For a RecipeCollection, it prints to a file all of the cuisine types
	 * and the number of recipes of that type.
	 * 
	 * @param filePath 	Path of the file to write containing cuisine information.
	 */
	public void outputCuisineTypes(String filePath){
		

		// Extract the list of cuisine types and sory them
		CuisineType[] cuisineTypeList = CuisineType.values();
	    Arrays.sort(cuisineTypeList);
	    
	    // Print the Cuisine Information to a file
        try{
			BufferedWriter fileOut = new BufferedWriter(new FileWriter(filePath));
			
			for(int i = 0; i < cuisineTypeList.length; i++){
				// Separate each cuisine type by a new line
				if(i != 0){
					fileOut.write(",");
					fileOut.newLine();
				}
				// Output the cuisine type and total number of recipes of that type
				String cType = cuisineTypeList[i].name();
				fileOut.write(cType);
			}
			fileOut.write(";");
			fileOut.close(); //--- Close the file writing.

		}
		catch(IOException e){
			System.out.print("Error: Unable to write output file.");
		}
		
	}
	
	/**
	 * Prints a RecipeCollection's set of ingredients to a file.
	 * It is printed in descending order of the ingredient's prevalence.  
	 * Format of each line is:
	 * 
	 * ingredientName,TotalRecipeCount,numbRecipesOfCuisineType00,numbRecipesOfCuisineType01,numbRecipesOfCuisineType02,...
	 * 
	 * @param filePath	String - Path of the file to print.
	 */
	public void outputIngredients(String filePath){
		// Extract the list of ingredients
		ArrayList<Ingredient> ingredientList = new ArrayList<Ingredient>(allIngredients.values());

		// Sort the cuisine types
	    Collections.sort(ingredientList, new Ingredient.RecipeCountCompareDescending() );
	    
	    // Print the Cuisine Information to a file
        try{
			BufferedWriter fileOut = new BufferedWriter(new FileWriter(filePath));
			
			for(int i = 0; i < ingredientList.size(); i++){
				Ingredient ingredient = ingredientList.get(i);
				// Separate each cuisine type by a new line
				if(i != 0){
					fileOut.newLine();
				}
				// Output the cuisine type and total number of recipes of that type
				fileOut.write(ingredient.getName() + "," + ingredient.getTotalRecipeCount());
				
				int[] ingredientTypeCount = ingredient.getAllCuisineTypesCount();
				for(int j = 0; j < ingredientTypeCount.length; j++)
					fileOut.write("," + ingredientTypeCount[j]);
			}
			fileOut.close(); //--- Close the file writing.

		}
		catch(IOException e){
			System.out.print("Error: Unable to write output file.");
		}
	}
	
	
	/**
	 * Divides a RecipeCollection into a training and test set based off
	 * a specified split percentage (e.g. 2/3).
	 * 
	 * @param splitRatio	Between 0 and 1. Ratio of elements to go in the training set.
	 * @return				Array of RecipeCollection objects.  First element is the training set.  The second is the test set.
	 */
	public RecipeCollection[] performRecipeHoldoutSplit(double splitRatio){
		
		// Get all recipes in this collection
		ArrayList<Recipe> shuffledRecipes = new ArrayList<Recipe>(Arrays.asList(this.getRecipes()));
		Collections.shuffle(shuffledRecipes);

		int i;
		// Create the training and test sets
		RecipeCollection trainingSet = new RecipeCollection();
		RecipeCollection testSet = new RecipeCollection();
		for(i = 0; i < shuffledRecipes.size(); i++){
			if(i < splitRatio * shuffledRecipes.size())
				trainingSet.addRecipe(shuffledRecipes.get(i));
			else
				testSet.addRecipe(shuffledRecipes.get(i));
		}
		
		return new RecipeCollection[]{trainingSet, testSet};
			
	}

	
	
	/**
	 * 
	 * Prints the Recipe File to a text file.  This can be used
	 * to simplify the creation of training files so as a generic importer can
	 * be used.
	 * 
	 * @param filePath	Path to the recipe file
	 */
	public void print(String filePath){
		// 
		ArrayList<Recipe> recipeList = new ArrayList<Recipe>(allRecipes.values());
		RecipeCollection.printRecipesFile(filePath, recipeList);
	}
	
	
	/**
	 * Helper method to print a set of recipes to a file.
	 * 
	 * @param filePath 	String - Path to the file to be printed
	 * @param recipes	Collection<Recipe> List of recipes to print to a file.
	 */
	public static void printRecipesFile(String filePath, List<Recipe> recipes){
	    
	    // Print the Cuisine Information to a file
        try{
			BufferedWriter fileOut = new BufferedWriter(new FileWriter(filePath));
			
			// Put the beginning of the JSON file.
			fileOut.write("[");
			fileOut.newLine();
			
			for(int i = 0; i < recipes.size(); i++){
				// Separate each cuisine type by a new line
				if(i != 0){
					fileOut.write(",");
					fileOut.newLine();
				}
				// Output recipe information
				String[] recipeOutput = recipes.get(i).toString().split("\n");
				for( int j = 0;  j < recipeOutput.length; j++){
					if(j != 0) fileOut.newLine();
					fileOut.write(recipeOutput[j]);
				}
			}
			
			// Put the end of the JSON file then close it.
			fileOut.newLine();
			fileOut.write("]");
			fileOut.close();

		}
		catch(IOException e){
			System.out.print("Error: Unable to write output file for the recipes.");
		}
	}

	/** 
	 * 
	 * Extracts the recipes in the recipe collection.
	 * 
	 * @return List of recipes
	 */
	private Recipe[] getRecipes(){
		Recipe[] recipeArr = new Recipe[allRecipes.size()];
		Set<Integer> keySet = allRecipes.keySet();
		Integer[] keys = keySet.toArray(new Integer[keySet.size()]);
		for(int i = 0; i < keys.length; i++)
			recipeArr[i] = allRecipes.get(keys[i]);
		return recipeArr;
	}
	
	
	public ValueDifferenceMetricCompare getValueDifferenceMetricCompare(){return vdmCompare;}
	public WeightedOverlapCoefficient getWeightedOverlapCoefficient(){ return new WeightedOverlapCoefficient(); }
	public AmalgamatedWeightedOverlapCoefficient getAmalgamatedWeightedOverlapCoefficient(){ return new AmalgamatedWeightedOverlapCoefficient(); }
	
	/**
	 * 
	 * 
	 * 
	 * @param testRecipeCollection	- Collection of Recipes whose class label will be determined.
	 * @param k						- Value of "K" for K-Nearest Neighbors
	 * @param dist					- Object that defines how the distance between two recipes is calculated.
	 * @return
	 */
	public RecipeResult performKNearestNeighbor(RecipeCollection testRecipeCollection, int k, RecipeDistance dist, 
												boolean useWeightedDistance, boolean useClassProbabilityWeighting, 
												boolean printIncrementalResults){
	
		// Helper class for sorting on recipes.
		class RecipeWrapper implements Comparable<RecipeWrapper>{
			Recipe recipe;
			double distance;
			
			public RecipeWrapper(Recipe recipe, double distance){
				this.recipe = recipe;
				this.distance = distance;
			}
			
			public Recipe getRecipe(){ return recipe; }
			
			public double getDistance(){ return distance; };
			
			@Override
			public int compareTo(RecipeWrapper other){
				/*if(this.distance < other.distance) return -1;
				else if(this.distance == other.distance) return 0;
				else return 1;*/
				return Double.compare(this.distance, other.distance);
			}
			
		}
		
		RecipeCollection.RecipeResult results = new RecipeResult();
		Recipe[] testRecipes = testRecipeCollection.getRecipes();
		Recipe[] trainingRecipes = this.getRecipes();
		int correctClassifications = 0;
		int correctFirstOrSecondClassifications = 0;
		RecipeWrapper[] sortedRecipes = new RecipeWrapper[trainingRecipes.length];
		for(int i = 0; i < testRecipes.length; i++){
			// Calculate the inter-recipe distance for each recipe and then sort by distance 
			for(int j = 0; j < trainingRecipes.length; j++){
				// Determine the distance between 
				double recipeDistance = dist.compare(testRecipes[i], trainingRecipes[j]);
				sortedRecipes[j] = new RecipeWrapper(trainingRecipes[j], recipeDistance);
			}
			Arrays.sort(sortedRecipes);
			
			// Iterate through the sorted recipes and find the most common cuisine types
			double[] recipeCuisineScore = new double[CuisineType.values().length];
			for(int j = 0; j < k; j++){
				// Get the cuisine type for this recipe
				CuisineType type = CuisineType.valueOf(sortedRecipes[j].getRecipe().getCuisineType());
				if(useWeightedDistance){
					recipeCuisineScore[type.ordinal()] += 1.0 / Math.pow(sortedRecipes[j].getDistance(),2);
				}
				else{
					recipeCuisineScore[type.ordinal()]+=1.0;
				}
			}
			
			// If this feature is enabled, normalize the probabilities based off class score likelihood.
			if(useClassProbabilityWeighting){
				for(int j = 0; j < CuisineType.count(); j++){
					//if(cuisineTypeCount[j] > cuisineTypeCount[maxId]) maxId = j;
					CuisineType cuisine = CuisineType.fromInt(j);
					recipeCuisineScore[j] /= cuisineTypeCount.get(cuisine);
				}
			}
			
			// Find the name of the cuisine types with the highest two scores.
			int maxId = 0;
			for(int j = 1; j < CuisineType.count(); j++){
				if(recipeCuisineScore[j] > recipeCuisineScore[maxId]) maxId = j;
			}
			int secondMaxId = (maxId!=0)?0:1;
			for(int j = secondMaxId + 1; j < CuisineType.count(); j++){
				if(recipeCuisineScore[j] > recipeCuisineScore[secondMaxId] && j!=maxId) secondMaxId = j;
			}
				
			// Get the name of the cuisine type that corresponds with this ordinal number
			String firstCuisineType = CuisineType.fromInt(maxId).name();
			String secondCuisineType = CuisineType.fromInt(secondMaxId).name();
			
			// Check if the classification is correct
			if(firstCuisineType.equals(testRecipes[i].getCuisineType())){
				correctClassifications++;
				correctFirstOrSecondClassifications++;
			}
			// Check if the second best selection is correct.
			else if(secondCuisineType.equals(testRecipes[i].getCuisineType())){
				correctFirstOrSecondClassifications++;
			}
			
			if(printIncrementalResults){
				if(i > 0 && i % INCREMENTAL_PRINT_FREQUENCY == 0){
					System.out.println(i + " out of " + testRecipes.length + " have been completed.");
					System.out.println("First place accuracy so far: " + String.format("%9.2f", 100.0 * correctClassifications/(i+1)) + "%.");
					System.out.println("Two two accuracy so far: " + String.format("%9.2f", 100.0 * correctFirstOrSecondClassifications/(i+1)) + "%.\n\n");
				}
			}
			
			// Normalize the cuisine score between 0 and 1
			double[] normalizedKNNScore = new double[recipeCuisineScore.length];
			double totalKNNScore = 0;
			for(int cuisineCnt = 0; cuisineCnt < recipeCuisineScore.length; cuisineCnt++)
				totalKNNScore += recipeCuisineScore[cuisineCnt];
			for(int cuisineCnt = 0; cuisineCnt < recipeCuisineScore.length; cuisineCnt++)
				normalizedKNNScore[cuisineCnt] = recipeCuisineScore[cuisineCnt] /totalKNNScore;
			results.addTestResult(testRecipes[i].getID(), normalizedKNNScore);
		}
		
		// Calculate the overall accuracy
		results.setAccuracy((double)correctClassifications/testRecipes.length);
		results.setTopTwoAccuracy((double)correctFirstOrSecondClassifications/testRecipes.length);

		return results;
		
	}

	
	
	
	public RecipeResult performKNNandBayesEnsemble(RecipeResult[] inputResults, boolean printIncrementalResults){
		
		Recipe[] testRecipes = this.getRecipes();
		RecipeCollection.RecipeResult results = new RecipeResult();
		int correctClassifications = 0;
		int correctFirstOrSecondClassifications = 0;
		
		
		for(int i = 0; i < testRecipes.length; i++){
			
			// Get the info on the test recipe
			int recipeID = testRecipes[i].getID();
			
			// Determine the combined scores
			double[] combinedScores = new double[CuisineType.count()];
			for(int resultCnt = 0; resultCnt < inputResults.length; resultCnt++){
				double[] tempScores = inputResults[resultCnt].getTestResult(recipeID); 
				for(int cuisineCnt = 0; cuisineCnt < CuisineType.count(); cuisineCnt++)
					combinedScores[cuisineCnt] += tempScores[cuisineCnt];
			}
			
			// Find the name of the cuisine types with the highest two scores.
			int maxId = 0;
			for(int j = 1; j < CuisineType.count(); j++){
				if(combinedScores[j] > combinedScores[maxId]) maxId = j;
			}
			int secondMaxId = (maxId!=0)?0:1;
			for(int j = secondMaxId + 1; j < CuisineType.count(); j++){
				if(combinedScores[j] > combinedScores[secondMaxId] && j!=maxId) secondMaxId = j;
			}
				
			// Get the name of the cuisine type that corresponds with this ordinal number
			String firstCuisineType = CuisineType.fromInt(maxId).name();
			String secondCuisineType = CuisineType.fromInt(secondMaxId).name();
			
			// Check if the classification is correct
			if(firstCuisineType.equals(testRecipes[i].getCuisineType())){
				correctClassifications++;
				correctFirstOrSecondClassifications++;
			}
			// Check if the second best selection is correct.
			else if(secondCuisineType.equals(testRecipes[i].getCuisineType())){
				correctFirstOrSecondClassifications++;
			}
			
			if(printIncrementalResults){
				if(i > 0 && i % INCREMENTAL_PRINT_FREQUENCY == 0){
					System.out.println(i + " out of " + testRecipes.length + " have been completed.");
					System.out.println("First place accuracy so far: " + String.format("%9.2f", 100.0 * correctClassifications/(i+1)) + "%.");
					System.out.println("Two two accuracy so far: " + String.format("%9.2f", 100.0 * correctFirstOrSecondClassifications/(i+1)) + "%.\n\n");
				}
			}
		}
		
		// Calculate the overall accuracy
		results.setAccuracy((double)correctClassifications/testRecipes.length);
		results.setTopTwoAccuracy((double)correctFirstOrSecondClassifications/testRecipes.length);

		return results;
		
	}
	
	
	
	
	/**
	 * 
	 * 
	 * 
	 * @param testRecipeCollection	- Collection of Recipes whose class label will be determined.
	 * @param k						- Value of "K" for K-Nearest Neighbors
	 * @param dist					- Object that defines how the distance between two recipes is calculated.
	 * @return
	 */
	public RecipeResult performNaiveBayes(RecipeCollection testRecipeCollection, 
										  IngredientConditionalProbability condProbability,
										  boolean printIncrementalResults){
		
		
		
		/*// Determine P(C) for each cuisine type
		double[] cuisineTypeProbability = new double[CuisineType.count()];
		CuisineType tempCT;
		for(int cnt = 0; cnt < CuisineType.count(); cnt++){
			tempCT = CuisineType.fromInt(cnt);
			cuisineTypeProbability[cnt] = ((float)this.cuisineTypeCount.get(tempCT)) / this.cuisineTypeCount.size();
		}*/
		
		RecipeResult results = new RecipeCollection.RecipeResult();
		Recipe[] testRecipes = testRecipeCollection.getRecipes();
		int correctClassifications = 0, correctFirstOrSecondClassifications = 0;
		Hashtable<String, ArrayList<Double>> allIngredientsClassProbabilities = new Hashtable<String, ArrayList<Double>>();
		
		// For each recipe, determine it the class probability
		Ingredient ingredient;
		for(int i = 0; i < testRecipes.length; i++){
			
			Recipe tempRecipe = testRecipes[i];
			String ingredientList[] = tempRecipe.getIngredients();
			double[] recipeBayesProbability = new double[CuisineType.count()]; // Define the recipe Bayes probability for each cuisine type
			
			// Start with the Bayes Probability equal to the cuisine type probability
			CuisineType tempCT;
			for(int cuisineCnt = 0; cuisineCnt < CuisineType.count();  cuisineCnt++){
				//recipeBayesProbability[cuisineCnt] = cuisineTypeProbability[cuisineCnt];
				tempCT = CuisineType.fromInt(cuisineCnt);
				recipeBayesProbability[cuisineCnt] = this.cuisineTypeCount.get(tempCT);
			}
				
			// Go through all the ingredients
			for(int ingredCnt = 0; ingredCnt < ingredientList.length; ingredCnt++){		
				String ingredName = ingredientList[ingredCnt];
				
				// Ensure this ingredient exists in the training set.  Otherwise skip it.
				ingredient = this.allIngredients.get(ingredName);
				if(ingredient == null) continue;
				
				// Check if the class probabilities for this ingredient have been calculated.
				ArrayList<Double> ingredientClassProbability = allIngredientsClassProbabilities.get(ingredName);
				if(ingredientClassProbability == null){
					
					// Calculate the class conditional probability.
					ingredientClassProbability = new ArrayList<Double>();
					for(int cuisineCnt = 0; cuisineCnt < CuisineType.count();  cuisineCnt++){
						// Get the cuisine type
						CuisineType type = CuisineType.fromInt(cuisineCnt);
						// Calculate the conditional class probability for this ingredient/cuisine type combination
						double tempProb = condProbability.calculate(ingredient.getCuisineTypeCount(cuisineCnt), 
																	cuisineTypeCount.get(type));
						// Store the conditional probability
						ingredientClassProbability.add(tempProb);
					}
					
					allIngredientsClassProbabilities.put(ingredName, ingredientClassProbability);
				}
				
				// Multiply each class by P(Ingredient | Class)
				for (int cuisineCnt = 0; cuisineCnt < CuisineType.count(); cuisineCnt++)
					recipeBayesProbability[cuisineCnt] *= ingredientClassProbability.get(cuisineCnt);
					
			}
			
			
			// Find the name of the cuisine types with the highest two scores.
			int maxId = 0;
			for(int j = 1; j < CuisineType.count(); j++){
				if(recipeBayesProbability[j] > recipeBayesProbability[maxId]) maxId = j;
			}
			int secondMaxId = (maxId!=0)?0:1;
			for(int j = secondMaxId + 1; j < CuisineType.count(); j++){
				if(recipeBayesProbability[j] > recipeBayesProbability[secondMaxId] && j!=maxId) secondMaxId = j;
			}
				
			// Get the name of the cuisine type that corresponds with this ordinal number
			String firstCuisineType = CuisineType.fromInt(maxId).name();
			String secondCuisineType = CuisineType.fromInt(secondMaxId).name();
			
			// Check if the classification is correct
			if(firstCuisineType.equals(testRecipes[i].getCuisineType())){
				correctClassifications++;
				correctFirstOrSecondClassifications++;
			}
			// Check if the second best selection is correct.
			else if(secondCuisineType.equals(testRecipes[i].getCuisineType())){
				correctFirstOrSecondClassifications++;
			}
			
			if(printIncrementalResults){
				if(i > 0 && i % INCREMENTAL_PRINT_FREQUENCY == 0){
					System.out.println(i + " out of " + testRecipes.length + " have been completed.");
					System.out.println("First place accuracy so far: " + String.format("%9.2f", 100.0 * correctClassifications/(i+1)) + "%.");
					System.out.println("Two two accuracy so far: " + String.format("%9.2f", 100.0 * correctFirstOrSecondClassifications/(i+1)) + "%.\n\n");
				}
			}
			
			// Normalize the Bayesian probability between 0 and 1
			double[] normalizedBayesScore = new double[recipeBayesProbability.length];
			double totalBayesProbability = 0;
			for(int cuisineCnt = 0; cuisineCnt < recipeBayesProbability.length; cuisineCnt++)
				totalBayesProbability += recipeBayesProbability[cuisineCnt];
			for(int cuisineCnt = 0; cuisineCnt < recipeBayesProbability.length; cuisineCnt++)
				normalizedBayesScore[cuisineCnt] = recipeBayesProbability[cuisineCnt] /totalBayesProbability;
			results.addTestResult(tempRecipe.getID(), normalizedBayesScore);
		}
		
		// Calculate the overall accuracy
		results.setAccuracy((double)correctClassifications/testRecipes.length);
		results.setTopTwoAccuracy((double)correctFirstOrSecondClassifications/testRecipes.length);

		return results;
		
	}
	
	
	
	
	

	public class RecipeResult {
	
		double accuracy;
		double topTwoAccuracy;// This is the combined accuracy of the first and second choice.
		
		public Hashtable<Integer, double[]> testResult = new Hashtable<Integer, double[]>();
		
		
		public double getAccuracy(){ return accuracy; }
		public void setAccuracy(double accuracy){ this.accuracy = accuracy; }
		
		public double getTopTwoAccuracy(){ return topTwoAccuracy; }
		public void setTopTwoAccuracy(double topTwoAccuracy){ this.topTwoAccuracy = topTwoAccuracy; }
		
		public void addTestResult(int recipeID, double[] results){ testResult.put(new Integer(recipeID), results); }
		public double[] getTestResult(int recipeID){ return testResult.get(recipeID); }
		
	}


	
	/**
	 * 
	 * This interface is used to allow one to specify different
	 * possible algorithms for calculating the distance between two recipes.
	 * 
	 * It is intended as a use of the STRATEGY PATTERN.  For more information on the 
	 * strategy pattern, see below:
	 * 
	 * https://en.wikipedia.org/wiki/Strategy_pattern
	 *
	 */
	public interface RecipeDistance{
		/**
		 * 
		 * Calculates the distance between r1 and r2.  The implementation
		 * is intended to be associative.
		 * 
		 * @param r1 A recipe
		 * @param r2 Another recipe
		 * @return The distance (i.e. similarity between r1 and r2.  The lower the
		 * return value, the more similar the two recipes are.  If two recipes 
		 * are identical, the return value should be 0.
		 * 
		 */
		double compare(Recipe r1, Recipe r2);
	}
	
	
	
	/**
	 * 
	 * This interface type is used to allow for different approaches 
	 * to calculate the conditional probability of a given ingredient.
	 *
	 */
	public interface IngredientConditionalProbability{
		
		/**
		 * 
		 * @param attributeClassRecordCount	Total number of records from the specified class with this attribute value.
		 * @param classRecordCount			Total number of records of the specified class.
		 * @return							P(A|C) in double form. 
		 */
		double calculate(int attributeClassRecordCount, int classRecordCount);
		
	}
	
	
	
	/**
	 * 
	 * This class is uses the accuracy metric to determine the conditional
	 * class probability of the ingredient. 
	 *
	 * Conditional Probability for this Class is the simplest and is:
	 * 
	 * P(A|C) = attributeClassRecordCount / classRecordCount
	 *
	 */
	public static class AccuracyIngredientClassProbability implements IngredientConditionalProbability{
		
		@Override
		public double calculate(int attributeClassRecordCount, int classRecordCount){
			return ((double)attributeClassRecordCount)/classRecordCount;
		}
		
	}
	
	
	/**
	 * 
	 * This class is uses the Laplace metric to determine the conditional
	 * class probability of the ingredient. 
	 * 
	 * Conditional Probability for this class is:
	 * 
	 * P(A|C) = (attributeClassRecordCount + 1) / (classRecordCount + numbCuisineTypes)
	 *
	 */
	public static class LaplaceIngredientClassProbability implements IngredientConditionalProbability{
		
		@Override
		public double calculate(int attributeClassRecordCount, int classRecordCount){
			return ((double)attributeClassRecordCount +1 )/(classRecordCount + CuisineType.count());
		}
		
	}
	
	
	/**
	 * 
	 * Used as a STRATEGY Method for determining the difference between
	 * two recipes in the collection.
	 *
	 */
	public static class OverlapCoefficient implements RecipeDistance{
		
		@Override
		public double compare(Recipe r1, Recipe r2){
			
			String[] r1Ingredients = r1.getIngredients();
			String[] r2Ingredients = r2.getIngredients();
			int totalMismatches = Math.min(r1Ingredients.length, r2Ingredients.length);
			
			// Iterate through each ingredient list
			for(int i = 0; i < r1Ingredients.length; i++){
				for(int j = 0; j < r2Ingredients.length; j++){
					if(r1Ingredients[i].equals(r2Ingredients[j])){
						totalMismatches--; // Remove one mismatch
						break;
					}
				}
			}
			
			// Normalize the distance to recipe length.
			return 1.0* totalMismatches/Math.min(r1Ingredients.length, r2Ingredients.length);
			
		}
		
	}
	
	/**
	 * 
	 * Used as a STRATEGY Method for determining the difference between
	 * two recipes in the collection.
	 *
	 */
	public static class OverlapCoefficientAmalgamated implements RecipeDistance{
		
		@Override
		public double compare(Recipe r1, Recipe r2){
			
			String[] r1Ingredients = r1.getIngredients();
			String[] r2Ingredients = r2.getIngredients();
			int totalMismatches = (r1Ingredients.length + r2Ingredients.length);
			
			// Iterate through each ingredient list
			for(int i = 0; i < r1Ingredients.length; i++){
				for(int j = 0; j < r2Ingredients.length; j++){
					if(r1Ingredients[i].equals(r2Ingredients[j])){
						totalMismatches--; // Remove one mismatch
						break;
					}
				}
			}
			
			// Normalize the distance to recipe length.
			return 1.0* totalMismatches/(r1Ingredients.length + r2Ingredients.length);
			
		}
		
	}
	
	
	/**
	 * 
	 * Used as a STRATEGY Method for determining the difference between
	 * two recipes in the collection.
	 * 
	 * This is a modified version of the standard OverlapCoefficient above. 
	 * This gives a weighted overlap bonus based off 1 / (1 + IngredientEntropy).
	 * 
	 * Hence an ingredient that appears in only one cuisine type that is matched will
	 * have its mismatch score reduced by one.  In contrast, an ingredient that is 
	 * in a lot of different recipe types will have the a lower mismatch reduction.
	 *
	 */
	public class WeightedOverlapCoefficient implements RecipeDistance{
		
		@Override
		public double compare(Recipe r1, Recipe r2){
			
			String[] r1Ingredients = r1.getIngredients();
			String[] r2Ingredients = r2.getIngredients();
			Ingredient ingredient;
			
			// Iterate through each ingredient list
			double totalMatches = 0;
			for(int i = 0; i < r1Ingredients.length; i++){
				for(int j = 0; j < r2Ingredients.length; j++){
					if(r1Ingredients[i].equals(r2Ingredients[j])){
						ingredient = allIngredients.get(r1Ingredients[i]);
						if(ingredient == null) continue;
						totalMatches += 1.0 / (1.0 + ingredient.getEntropy()); // Remove one mismatch
						break;
					}
				}
			}
			
			// Calculate the minimum length between the two recipes
			double r1Length = 0;
			for(int i=0; i < r1Ingredients.length; i++){
				ingredient = allIngredients.get(r1Ingredients[i]);
				if(ingredient == null) continue;
				r1Length += 1.0 / (1.0 + ingredient.getEntropy()); // Remove one mismatch
			}
			double r2Length = 0;
			for(int i=0; i < r2Ingredients.length; i++){
				ingredient = allIngredients.get(r2Ingredients[i]);
				if(ingredient == null) continue;
				r2Length += 1.0 / (1.0 + ingredient.getEntropy()); // Remove one mismatch
			}
			double minLength =Math.min(r1Length,  r2Length);
			minLength = Math.min(minLength, 0.1);// Prevent a divide by zero.
			
			// Normalize the distance to recipe length.
			return (minLength - totalMatches)/minLength;
			
		}
		
	}
	
	
	public class AmalgamatedWeightedOverlapCoefficient implements RecipeDistance{
		
		@Override
		public double compare(Recipe r1, Recipe r2){
			
			String[] r1Ingredients = r1.getIngredients();
			String[] r2Ingredients = r2.getIngredients();
			Ingredient ingredient;
			
			// Iterate through each ingredient list
			double totalMatches = 0;
			for(int i = 0; i < r1Ingredients.length; i++){
				for(int j = 0; j < r2Ingredients.length; j++){
					if(r1Ingredients[i].equals(r2Ingredients[j])){
						ingredient = allIngredients.get(r1Ingredients[i]);
						if(ingredient == null) continue;
						totalMatches += 1.0 / (1.0 + ingredient.getEntropy()); // Remove one mismatch
						break;
					}
				}
			}
			
			// Calculate the minimum length between the two recipes
			double r1Length = 0;
			for(int i=0; i < r1Ingredients.length; i++){
				ingredient = allIngredients.get(r1Ingredients[i]);
				if(ingredient == null) continue;
				r1Length += 1.0 / (1.0 + ingredient.getEntropy()); // Remove one mismatch
			}
			double r2Length = 0;
			for(int i=0; i < r2Ingredients.length; i++){
				ingredient = allIngredients.get(r2Ingredients[i]);
				if(ingredient == null) continue;
				r2Length += 1.0 / (1.0 + ingredient.getEntropy()); // Remove one mismatch
			}
			double totalLength = r1Length + r2Length;
			totalLength = Math.min(totalLength, 0.0000001);// Prevent a divide by zero.
			
			// Normalize the distance to recipe length.
			return (totalLength - totalMatches)/totalLength;
			
		}
		
	}
	
	
	
	/**
	 * 
	 * Used as a STRATEGY Method for determining the difference between
	 * two recipes in the collection.
	 *
	 */
	public class ValueDifferenceMetricCompare implements RecipeDistance{
		
		Hashtable<String, Double> ingredientDistance = new Hashtable<String, Double>();
		
		private final static int  MIN_INGREDIENT_PAIRS = 5;
		
		@Override
		public double compare(Recipe r1, Recipe r2){

			double totalDistance = 0;
			double calculatedDistance;
			Double storedDistance;
			String i1Name, i2Name;
			Ingredient i1, i2;
			String[] r1Ingredients = r1.getIngredients();
			String[] r2Ingredients = r2.getIngredients();
			int illegalIngredientPairs = 0; // This is caused by an ingredient from either recipe not being in the known set
			
			// Iterate through each ingredient list
			for(int i = 0; i < r1Ingredients.length; i++){
				
				i1Name = r1Ingredients[i];
				// Check if i1 is a known ingredient name
				i1 = allIngredients.get(i1Name);
				if(i1 == null){
					illegalIngredientPairs += r2Ingredients.length;
					continue;
				}
				
				for(int j = 0; j < r2Ingredients.length; j++){
					i2Name = r2Ingredients[j];
					
					// If the ingredients are identical, go to the next ingredient.
					if(i1Name.equals(i2Name)) continue;
					
					// Check if i2 is a known ingredient name
					i2 = allIngredients.get(i2Name);
					if(i2 == null){
						illegalIngredientPairs++;
						continue;
					}
					
					// See if the ingredient distance is stored
					storedDistance = ingredientDistance.get(getKeyName(i1Name, i2Name));
					if(storedDistance != null){
						totalDistance += storedDistance.doubleValue();
						continue;
					}
					
					// Ingredient distance is not stored so calculate it.
					calculatedDistance = 0.0;
					int[] i1CuisineCounts = i1.getAllCuisineTypesCount();
					int[] i2CuisineCounts = i2.getAllCuisineTypesCount();
					for(int k = 0; k < CuisineType.count(); k++){
						calculatedDistance += Math.abs((1.0 * i1CuisineCounts[k])/i1.getTotalRecipeCount() 
												       - (1.0*i2CuisineCounts[k])/i2.getTotalRecipeCount());
					}
					// Store the inter-ingredient distance in the collection.
					storedDistance = new Double(calculatedDistance); // Convert to a Wrapper Double object (i.e. non-double primitive)
					ingredientDistance.put(getKeyName(i1Name, i2Name), storedDistance);
					totalDistance += calculatedDistance;
				}
			}
			
			// Normalize the distance to recipe length.
			if(r1Ingredients.length * r2Ingredients.length - illegalIngredientPairs > MIN_INGREDIENT_PAIRS)
				totalDistance /= (r1Ingredients.length * r2Ingredients.length - illegalIngredientPairs);
			else
				totalDistance = Double.MAX_VALUE;
			// Return the total distance
			return totalDistance;
			
		}
		
		/**
		 * 
		 * Builds a key given two ingredient names
		 * 
		 * @param i1 Name of the first ingredient
		 * @param i2 Name of the second ingredient
		 * 
		 * @return Hash table key name for two ingredients.
		 */
		private String getKeyName(String i1, String i2){
			
			// If i1 is alphabetically before i2, then it is the first in the key.
			if(i1.compareTo(i2) < 0) return i1 + "_" + i2;
			else 					 return i2 + "_" + i1;
			
			
		}
		
	}

}


