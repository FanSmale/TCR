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
	}//Of the first constructor
	
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
			// System.out.println("paraUser = " + paraUser + ", i = " + i + ",
			// tempItemIndex = " + tempItemIndex);
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
		for (i = 0; i < trainingIndicationMatrix[paraUser].length; i++) {
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
	public void train(int paraUser, int paraRounds) {
		for (int i = 0; i < paraRounds; i++) {
			update(paraUser);
		} // Of for i
	}// Of train
	
	/**
	 ************************ 
	 * Update sub-spaces using the training data.
	 ************************ 
	 */
	public void update(int paraUser) {
		switch (regularScheme) {
		case NO_REGULAR:
			updateNoRegular(paraUser);
			break;
		case PQ_REGULAR:
			updatePQRegular(paraUser);
			break;
		default:
			System.out.println("Unsupported regular scheme: " + regularScheme);
			System.exit(0);
		}// Of switch
	}// Of update

	/**
	 ************************ 
	 * Update sub-spaces using the training data.
	 ************************ 
	 */
	public void updateNoRegular(int paraUser) {
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

			// Update item subspace
			for (int j = 0; j < rank; j++) {
				tempValue = 2 * tempResidual * userSubspace[paraUser][j];

				itemSubspace[tempItemId][j] += alpha * tempValue;
			} // Of for j
		} // Of for i
	}// Of updateNoRegular

	/**
	 ************************ 
	 * Update sub-spaces using the training data.
	 ************************ 
	 */
	public void updatePQRegular(int paraUser) {
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

			// Update item subspace
			for (int j = 0; j < rank; j++) {
				tempValue = 2 * tempResidual * userSubspace[paraUser][j] - lambda * itemSubspace[tempItemId][j];
				itemSubspace[tempItemId][j] += alpha * tempValue;
			} // Of for j
		} // Of for i
	}// Of updatePQRegular
	
	/**
	 ************************ 
	 * The training testing scenario.
	 ************************ 
	 */
	public static void testIncremental(String paraFilename, int paraNumUsers, int paraNumItems, int paraNumRatings,
			double paraRatingLowerBound, double paraRatingUpperBound) {
		// Step 1. Read data and set parameters.
		MF2DBooleanIncremental tempMF = null;
		try {
			tempMF = new MF2DBooleanIncremental(paraFilename, paraNumUsers, paraNumItems, paraNumRatings, paraRatingLowerBound,
					paraRatingUpperBound);
		} catch (Exception e) {
			e.printStackTrace();
		} // Of try

		tempMF.setParameters(10, 0.0001, 0.005, PQ_REGULAR);
		tempMF.setAllTraining();
		tempMF.adjustUsingMeanRating();

		// Step 2. Pre-train
		tempMF.initializeSubspaces(0.5);
		System.out.println("Pre-training 200 rounds ...");
		tempMF.train(100);

		// Step 3. Train for each user
		double tempMAE;
		int tempNumItemsForTrain = 0;
		int tempNumPredictions = 0;
		double tempErrorSum = 0;
		for (int i = 0; i < tempMF.numUsers; i++) {
			System.out.println("User " + i);

			// Step 3.1 One half items, e.g., {0, 2, 4, ...} for training.
			tempNumItemsForTrain = tempMF.data[i].length / 2;
			int[] tempIndices = new int[tempNumItemsForTrain];
			for (int j = 0; j < tempNumItemsForTrain; j++) {
				tempIndices[j] = tempMF.data[i][j * 2].item;
			} // Of for j

			// System.out.println("tempIndices = " +
			// Arrays.toString(tempIndices));
			tempMF.setUserTraining(i, tempIndices);

			// Step 3.2 Incremental training.
			//tempMF.train(i, 10);

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
			//tempMF.train(i, 10);

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
		testIncremental("data/jester-data-1/jester-data-1.txt", 24983, 101, 1810455, -10, 10);
	}// Of main
}//Of class MF2DBooleanIncremental