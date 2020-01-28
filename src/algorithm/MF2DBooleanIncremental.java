package algorithm;

/*
 * @(#)MF2DBoolean.java
 
 * Project: Matrix factorization for recommender systems.
 * Incremental learning. 
 * Author: Fan Min, Yuan-Yuan Xu
 * www.fansmale.com
 * Email: minfan@swpu.edu.cn, minfanphd@163.com.
 * Created: Jan 21, 2020.
 * Last modified: Jan 21, 2020.
 */

public class MF2DBooleanIncremental extends MF2DBoolean {
	/**
	 * Incremental training rounds.
	 */
	int incrementalTrainRounds = 20;

	/**
	 ************************ 
	 * The first constructor.
	 * 
	 * @param paraFilename
	 *            The data filename.
	 * @param paraNumUsers
	 *            The number of users.
	 * @param paraNumItems
	 *            The number of items.
	 * @param paraNumRatings
	 *            The number of ratings.
	 ************************ 
	 */
	public MF2DBooleanIncremental(String paraFilename, int paraNumUsers, int paraNumItems, int paraNumRatings,
			double paraRatingLowerBound, double paraRatingUpperBound) {
		super(paraFilename, paraNumUsers, paraNumItems, paraNumRatings, paraRatingLowerBound, paraRatingUpperBound);
	}// Of the first constructor
	
	/**
	 ************************ 
	 * Setter.
	 ************************ 
	 */
	public void setIncrementalTrainRounds(int paraValue){
		incrementalTrainRounds = paraValue;
	}//Of setIncrementalTrainRounds

	/**
	 ************************ 
	 * Set all data of the given user for training.
	 * 
	 * @param paraUser
	 *            The given user.
	 ************************ 
	 */
	public void setUserAllTraining(int paraUser) {
		for (int i = 0; i < trainingIndicationMatrix[paraUser].length; i++) {
			trainingIndicationMatrix[paraUser][i] = true;
		} // Of for j
	}// Of setUserAllTraining

	/**
	 ************************ 
	 * Set some data of the given user for training.
	 * 
	 * @param paraUser
	 *            The given user.
	 * @param paraTrainingIndices
	 *            The item indices for the given user as training.
	 ************************ 
	 */
	public void setUserTraining(int paraUser, int[] paraTrainingItems) {
		int tempItemIndex = 0;
		int i;
		for (i = 0; i < trainingIndicationMatrix[paraUser].length; i++) {
			if (data[paraUser][i].item == paraTrainingItems[tempItemIndex]) {
				trainingIndicationMatrix[paraUser][i] = true;
				tempItemIndex++;
				if (tempItemIndex == paraTrainingItems.length) {
					break;
				} // Of if
			} else {
				trainingIndicationMatrix[paraUser][i] = false;
			} // Of if
		} // Of for j

		// The remaining parts are all testing.
		//Attention: i should not be re-initialized!
		for (; i < trainingIndicationMatrix[paraUser].length; i++) {
			trainingIndicationMatrix[paraUser][i] = false;
		} // Of for i
	}// Of setUserTraining

	/**
	 ************************ 
	 * Train according to data of the user.
	 * 
	 * @param paraUser
	 *            The given user.
	 * @param paraRounds
	 *            The number of rounds.
	 ************************ 
	 */
	public void trainUser(int paraUser) {
		// Step 1. Reset the user subspace of the given user.
		for (int i = 0; i < rank; i++) {
			userSubspace[paraUser][i] += (rand.nextDouble() - 0.5) * 2 * subspaceValueRange;
		} // Of for i
		//System.out.println("initialize userSubspace[" + paraUser + "] = " + Arrays.toString(userSubspace[paraUser]));

		// Step 2. Update the user subspace.
		for (int i = 0; i < incrementalTrainRounds; i++) {
			updateUserSubspace(paraUser);
		} // Of for i
	}// Of trainUser

	/**
	 ************************ 
	 * Update sub-spaces using the training data.
	 ************************ 
	 */
	public void updateUserSubspace(int paraUser) {
		switch (regularScheme) {
		case NO_REGULAR:
			updateUserSubspaceNoRegular(paraUser);
			break;
		case PQ_REGULAR:
			updateUserSubspacePQRegular(paraUser);
			break;
		default:
			System.out.println("Unsupported regular scheme: " + regularScheme);
			System.exit(0);
		}// Of switch
	}// Of update

	/**
	 ************************ 
	 * Update the user sub-space using the training data of the given user.
	 * 
	 * @param paraUser
	 *            The given user.
	 ************************ 
	 */
	public void updateUserSubspaceNoRegular(int paraUser) {
		for (int i = 0; i < data[paraUser].length; i++) {
			// Ignore the testing set.
			if (!trainingIndicationMatrix[paraUser][i]) {
				continue;
			} // Of if

			int tempItemId = data[paraUser][i].item;
			double tempRate = data[paraUser][i].rating;

			double tempResidual = tempRate - predict(paraUser, tempItemId); // Residual
			// tempResidual = Math.abs(tempResidual);

			// Update user subspace
			double tempValue = 0;
			for (int j = 0; j < rank; j++) {
				tempValue = 2 * tempResidual * itemSubspace[tempItemId][j];
				userSubspace[paraUser][j] += alpha * tempValue;
			} // Of for j
			//System.out.println("i = " + i + ", userSubspace[" + paraUser + "] = " + Arrays.toString(userSubspace[paraUser]));
		} // Of for i
	}// Of updateUserSubspaceNoRegular	
	
	/**
	 ************************ 
	 * Update the user sub-space using the training data of the given user.
	 * 
	 * @param paraUser
	 *            The given user.
	 ************************ 
	 */
	public void updateUserSubspacePQRegular(int paraUser) {
		for (int i = 0; i < data[paraUser].length; i++) {
			// Ignore the testing set.
			if (!trainingIndicationMatrix[paraUser][i]) {
				continue;
			} // Of if

			int tempItemId = data[paraUser][i].item;
			double tempRate = data[paraUser][i].rating;

			double tempResidual = tempRate - predict(paraUser, tempItemId); // Residual
			// tempResidual = Math.abs(tempResidual);

			// Update user subspace
			double tempValue = 0;
			for (int j = 0; j < rank; j++) {
				tempValue = 2 * tempResidual * itemSubspace[tempItemId][j] - lambda * userSubspace[paraUser][j];
				userSubspace[paraUser][j] += alpha * tempValue;
			} // Of for j
		} // Of for i
		//System.out.println("PQ regular: " + Arrays.toString(userSubspace[paraUser]));
	}// Of updateUserSubspacePQRegular

	/**
	 ************************ 
	 * Pre-train. All data are employed for training.
	 ************************ 
	 */
	public void pretrain() {
		//setParameters(10, 0.0001, 0.005, NO_REGULAR, paraRounds);
		setAllTraining();
		adjustUsingMeanRating();

		// Step 2. Pre-train
		initializeSubspaces(0.5);
		//System.out.println("Pre-training " + paraRounds + " rounds ...");
		train();
	}// Of pretrain

	/**
	 ************************ 
	 * The training testing scenario.
	 ************************ 
	 */
	public static void testIncremental(String paraFilename, int paraNumUsers, int paraNumItems, int paraNumRatings,
			double paraRatingLowerBound, double paraRatingUpperBound, int paraRounds, int paraIncrementalRounds) {
		// Step 1. Read data and set parameters.
		MF2DBooleanIncremental tempMF = null;
		try {
			tempMF = new MF2DBooleanIncremental(paraFilename, paraNumUsers, paraNumItems, paraNumRatings,
					paraRatingLowerBound, paraRatingUpperBound);
		} catch (Exception e) {
			e.printStackTrace();
		} // Of try

		tempMF.setParameters(10, 0.0001, 0.005, NO_REGULAR, 200);
		tempMF.setAllTraining();
		tempMF.adjustUsingMeanRating();

		// Step 2. Pre-train
		tempMF.initializeSubspaces(0.5);
		System.out.println("Pre-training " + paraRounds + " rounds ...");
		tempMF.train(paraRounds);

		// Step 3. Train for each user
		double tempMAE;
		int tempNumItemsForTrain = 0;
		int tempNumPredictions = 0;
		double tempErrorSum = 0;
		for (int i = 0; i < tempMF.numUsers; i++) {
			//System.out.println("User " + i);

			// Step 3.1 One half items, e.g., {0, 2, 4, ...} for training.
			tempNumItemsForTrain = tempMF.data[i].length / 2;
			int[] tempIndices = new int[tempNumItemsForTrain];
			for (int j = 0; j < tempNumItemsForTrain; j++) {
				tempIndices[j] = tempMF.data[i][j * 2].item;
			} // Of for j
			tempMF.setUserTraining(i, tempIndices);
			
			// Step 3.2 Incremental training.
			tempMF.trainUser(i);

			// Step 3.3 Prediction and compute error.
			int tempItem;
			double tempPrediction;
			for (int j = tempNumItemsForTrain; j < tempMF.data[i].length; j++) {
				tempItem = tempMF.data[i][j].item;
				tempPrediction = tempMF.predict(i, tempItem);
				tempErrorSum += Math.abs(tempPrediction - tempMF.data[i][j].rating);
				tempNumPredictions++;
			} // Of for j

			// Step 3.4 Restore data of this user.
			tempMF.setUserAllTraining(i);

			// Step 3.5 Show message.
			tempMAE = tempErrorSum / tempNumPredictions;
			System.out.println("MAE = " + tempErrorSum + " / " + tempNumPredictions + " = " + tempMAE);
		} // Of for i

		tempMAE = tempErrorSum / tempNumPredictions;
		System.out.println("With incremental updating, MAE = " + tempMAE);
	}// Of testIncremental

	/**
	 ************************ 
	 * @param args
	 ************************ 
	 */
	public static void main(String args[]) {
		testIncremental("data/jester-data-1/jester-data-1.txt", 24983, 101, 1810455, -10, 10, 200, 100);
	}// Of main
}// Of class MF2DBooleanIncremental
