package algorithm;

import java.util.Arrays;

import common.*;
import datamodel.*;

/**
 * Incremental learning for matrix factorization, where data is stand alone.
 * <br>
 * Project: Three-way conversational recommendation.<br>
 * 
 * @author Fan Min<br>
 *         www.fansmale.com, github.com/fansmale/TCR.<br>
 *         Email: minfan@swpu.edu.cn, minfanphd@163.com.<br>
 * @date Created: December 3, 2019.<br>
 *       Last modified: January 30, 2020.
 * @version 1.0
 */

public class MF2DBooleanIncremental extends MF2DBoolean {
	/**
	 * Incremental training rounds.
	 */
	int incrementalTrainRounds = 20;

	/**
	 * Item predicted rating no less than the second value may be recommended,
	 * no less than the first value may be promoted.
	 */
	double[] favoriteThresholds = { -2.0, 0.5 };

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
	 * @param paraNumRatings
	 *            The number of ratings.
	 * @param paraRatingLowerBound
	 *            The lower bound of ratings.
	 * @param paraRatingUpperBound
	 *            The upper bound of ratings.
	 * @param paraCompress
	 *            Is the data in compress format?
	 ************************ 
	 */
	public MF2DBooleanIncremental(String paraFilename, int paraNumUsers, int paraNumItems,
			int paraNumRatings, double paraRatingLowerBound, double paraRatingUpperBound,
			boolean paraCompress) {
		super(paraFilename, paraNumUsers, paraNumItems, paraNumRatings, paraRatingLowerBound,
				paraRatingUpperBound, paraCompress);
	}// Of the first constructor

	/**
	 ************************ 
	 * The second constructor.
	 * 
	 * @param paraDataset
	 *            The given dataset.
	 ************************ 
	 */
	public MF2DBooleanIncremental(RatingSystem2DBoolean paraDataset) {
		super(paraDataset);
	}// Of the second constructor

	/**
	 ************************ 
	 * Setter.
	 ************************ 
	 */
	public void setIncrementalTrainRounds(int paraValue) {
		incrementalTrainRounds = paraValue;
	}// Of setIncrementalTrainRounds

	/**
	 *********************************** 
	 * Setter.
	 * 
	 * @param paraThresholds
	 *            Should have exactly two elements.
	 *********************************** 
	 */
	public void setFavoriteThresholds(double[] paraThresholds) {
		favoriteThresholds = paraThresholds;
	}// Of setFavoriteThresholds

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
			userSubspace[paraUser][i] += (Common.random.nextDouble() - 0.5) * 2
					* subspaceValueRange;
		} // Of for i
			// System.out.println("initialize userSubspace[" + paraUser + "] = "
			// + Arrays.toString(userSubspace[paraUser]));

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
		for (int i = 0; i < dataset.getUserNumRatings(paraUser); i++) {
			// Ignore the testing set.
			if (!dataset.getTrainIndication(paraUser, i)) {
				continue;
			} // Of if

			Triple tempTriple = dataset.getTriple(paraUser, i);
			int tempItemId = tempTriple.item;
			double tempRating = tempTriple.rating;

			double tempResidual = tempRating - predict(paraUser, tempItemId); // Residual
			// tempResidual = Math.abs(tempResidual);

			// Update user subspace
			double tempValue = 0;
			for (int j = 0; j < rank; j++) {
				tempValue = 2 * tempResidual * itemSubspace[tempItemId][j];
				userSubspace[paraUser][j] += alpha * tempValue;
			} // Of for j
				// System.out.println("i = " + i + ", userSubspace[" + paraUser
				// + "] = " + Arrays.toString(userSubspace[paraUser]));
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
		for (int i = 0; i < dataset.getUserNumRatings(paraUser); i++) {
			// Ignore the testing set.
			if (!dataset.getTrainIndication(paraUser, i)) {
				continue;
			} // Of if

			Triple tempTriple = dataset.getTriple(paraUser, i);
			int tempItemId = tempTriple.item;
			double tempRating = tempTriple.rating;

			double tempResidual = tempRating - predict(paraUser, tempItemId); // Residual
			// tempResidual = Math.abs(tempResidual);

			// Update user subspace
			double tempValue = 0;
			for (int j = 0; j < rank; j++) {
				tempValue = 2 * tempResidual * itemSubspace[tempItemId][j]
						- lambda * userSubspace[paraUser][j];
				userSubspace[paraUser][j] += alpha * tempValue;
			} // Of for j
		} // Of for i

		// System.out.println("PQ regular: " +
		// Arrays.toString(userSubspace[paraUser]));
	}// Of updateUserSubspacePQRegular

	/**
	 ************************ 
	 * Pre-train. All data are employed for training.
	 ************************ 
	 */
	public void pretrain() {
		// setParameters(10, 0.0001, 0.005, NO_REGULAR, paraRounds);
		dataset.setAllTraining();
		dataset.adjustUsingMeanRating();

		// Step 2. Pre-train
		initializeSubspaces(0.5);
		// System.out.println("Pre-training " + paraRounds + " rounds ...");
		train();
	}// Of pretrain

	/**
	 *************************
	 * One round three-way recommend according to existing recommendation
	 * information. These information will be changed in this method.
	 * 
	 * @param paraUser
	 *            The user.
	 * @param paraRecommendations
	 *            Indicate which items have already been recommended.
	 * @param paraPromotions
	 *            Indicate which items have already been promoted.
	 * @return An integer matrix, where the first row indicates recommended
	 *         items, while the second indicates promoted ones.
	 *************************
	 */
	public boolean[][] recommendForUser(int paraUser, boolean[] paraRecommendations,
			boolean[] paraPromotions) {
		double tempActualThreshold = likeThreshold - dataset.getMeanRating();
		while (true) {
			int[][] tempRecommendationsPromotions = threeWayRecommend(paraUser, paraRecommendations,
					paraPromotions);
			if (tempRecommendationsPromotions == null) {
				break;
			} // Of if

			boolean tempOneSuccess = false;
			double tempRating;
			for (int i = 0; i < tempRecommendationsPromotions.length; i++) {
				for (int j = 0; j < tempRecommendationsPromotions[i].length; j++) {
					tempRating = dataset.getUserItemRating(paraUser,
							tempRecommendationsPromotions[i][j]);
					if ((tempRating != RatingSystem2DBoolean.DEFAULT_MISSING_RATING)
							&& (tempRating > tempActualThreshold)) {
						SimpleTools.processTrackingOutput(
								"" + tempRecommendationsPromotions[i][j] + " successful.");
						tempOneSuccess = true;
						break;
					} // Of if
				} // Of for j
				if (tempOneSuccess) {
					break;
				} // Of if
			} // Of for i

			if (!tempOneSuccess) {
				break;
			} // Of if

			SimpleTools.processTrackingOutput("Recommend "
					+ Arrays.toString(tempRecommendationsPromotions[0]) + ", promote "
					+ Arrays.toString(tempRecommendationsPromotions[1]) + " to next round.");
		} // Of while

		boolean[][] resultRecommendationPromotions = new boolean[2][];
		resultRecommendationPromotions[0] = paraRecommendations;
		resultRecommendationPromotions[1] = paraPromotions;
		return resultRecommendationPromotions;
	}// Of recommendForUser

	/**
	 *************************
	 * One round three-way recommend according to existing recommendation
	 * information. These information will be changed in this method.
	 * 
	 * @param paraUser
	 *            The user.
	 * @param paraRecommendations
	 *            Indicate which items have already been recommended.
	 * @param paraPromotions
	 *            Indicate which items have already been promoted.
	 * @return An integer matrix, where the first row indicates recommended
	 *         items, while the second indicates promoted ones.
	 *************************
	 */
	public int[][] threeWayRecommend(int paraUser, boolean[] paraRecommendations,
			boolean[] paraPromotions) {

		int tempUserNumRates = dataset.getUserNumRatings(paraUser);
		int[] tempAcquiredItems = new int[tempUserNumRates];
		int tempCounter = 0;
		int[] tempRecommendationCandidates = new int[numItems];
		int[] tempPromotionCandidates = new int[numItems];

		// Step 1. Which items have rating information available.
		tempCounter = 0;
		for (int i = 0; i < tempUserNumRates; i++) {
			int tempItem = dataset.getTriple(paraUser, i).item;
			if (paraRecommendations[tempItem] || paraPromotions[tempItem]) {
				tempAcquiredItems[tempCounter] = tempItem;
				tempCounter++;
			} // Of if
		} // Of for i

		// Compress
		int[] tempCompressedItems = new int[tempCounter];
		for (int i = 0; i < tempCounter; i++) {
			tempCompressedItems[i] = tempAcquiredItems[i];
		} // Of for i

		if (tempCounter == 0) {
			System.out.println("Warning in MF2DBooleanIncremental.threeWayRecommend().\r\n"
					+ "No known ratings for  user #" + paraUser + "\r\n"
					+ "This may be caused by inappropriate popularity parameters for popularity-based recommendation.");
			// System.exit(0);
		} // Of if

		// Step 2. Predict for the current user.
		dataset.setUserTraining(paraUser, tempCompressedItems);
		double[] tempPredicts = predictForUser(paraUser);

		// Step 3. Generate recommendation/promotion candidates list
		int tempRecommendationCandidatesLength = 0;
		int tempPromotionCandidatesLength = 0;

		for (int i = 0; i < numItems; i++) {
			// Already recommended/promoted before
			if (paraRecommendations[i] || paraPromotions[i]) {
				continue;
			} // Of if

			// System.out.println("tempPredicts[" + i + "]= " + tempPredicts[i]
			// + "vs. "
			// + (favoriteThresholds[1] - dataset.getMeanRating()) + " and "
			// + (favoriteThresholds[0] - dataset.getMeanRating()));
			if (tempPredicts[i] >= favoriteThresholds[1] - dataset.getMeanRating()) {
				tempRecommendationCandidates[tempRecommendationCandidatesLength] = i;
				tempRecommendationCandidatesLength++;
				// System.out.println("May recommend " + i);
			} else if (tempPredicts[i] >= favoriteThresholds[0] - dataset.getMeanRating()) {
				tempPromotionCandidates[tempPromotionCandidatesLength] = i;
				tempPromotionCandidatesLength++;
				// System.out.println("May promote " + i);
			} // Of if
		} // Of for i

		// Step 4. Handle the situation where no enough to
		// recommend/promote.
		if (tempRecommendationCandidatesLength < numRecommend) {
			// System.out.println("User " + paraUser + " has no enough to
			// recommend: "
			// + tempRecommendationCandidatesLength);
			return null;
		} else if (tempPromotionCandidatesLength < numPromote) {
			// System.out.println("User " + paraUser + " has no enough to
			// promote: "
			// + tempPromotionCandidatesLength);
			return null;
		} // Of if

		// Step 5. Randomly select some to recommend/promote.
		int[] tempRecommendations = null;
		int[] tempPromotions = null;
		try {
			// Recommend
			tempRecommendations = SimpleTools.randomSelectFromArray(tempRecommendationCandidates,
					tempRecommendationCandidatesLength, numRecommend);
			for (int i = 0; i < tempRecommendations.length; i++) {
				paraRecommendations[tempRecommendations[i]] = true;
			} // Of for i

			// System.out.println("Recommend " +
			// Arrays.toString(tempRecommendations) + " to " + paraUser);

			// Promote
			tempPromotions = SimpleTools.randomSelectFromArray(tempPromotionCandidates,
					tempPromotionCandidatesLength, numPromote);
			for (int i = 0; i < tempPromotions.length; i++) {
				paraPromotions[tempPromotions[i]] = true;
			} // Of for i

			// System.out.println("Promote " + Arrays.toString(tempPromotions) +
			// " to " + paraUser);
		} catch (Exception ee) {
			System.out.println(
					"Error occurred in MF2DBooleanIncrementalAlone.threeWayRecommend(int)\r\n"
							+ ee);
		} // Of try

		// Step 5. Construct the lists.
		int[][] resultArrays = new int[2][];
		resultArrays[0] = tempRecommendations;
		resultArrays[1] = tempPromotions;

		return resultArrays;
	}// Of threeWayRecommend

	/**
	 ************************ 
	 * The training testing scenario.
	 ************************ 
	 */
	public static void testIncremental(String paraFilename, int paraNumUsers, int paraNumItems,
			int paraNumRatings, double paraRatingLowerBound, double paraRatingUpperBound,
			boolean paraCompress, int paraRounds, int paraIncrementalRounds) {
		// Step 1. Read data and set parameters.

		RatingSystem2DBoolean tempDataset = null;
		try {
			tempDataset = new RatingSystem2DBoolean(paraFilename, paraNumUsers, paraNumItems,
					paraNumRatings, paraRatingLowerBound, paraRatingUpperBound, paraCompress);
		} catch (Exception e) {
			e.printStackTrace();
		} // Of try
		tempDataset.setAllTraining();
		tempDataset.adjustUsingMeanRating();

		MF2DBooleanIncremental tempLearner = new MF2DBooleanIncremental(tempDataset);

		tempLearner.setParameters(10, 0.0001, 0.005, NO_REGULAR, 200);

		// Step 2. Pre-train
		tempLearner.initializeSubspaces(0.5);
		System.out.println("Pre-training " + paraRounds + " rounds ...");
		tempLearner.train(paraRounds);

		// Step 3. Train for each user
		double tempMAE;
		int tempNumItemsForTrain = 0;
		int tempNumPredictions = 0;
		double tempErrorSum = 0;
		for (int i = 0; i < tempDataset.getNumUsers(); i++) {
			// System.out.println("User " + i);

			// Step 3.1 One half items, e.g., {0, 2, 4, ...} for training.
			tempNumItemsForTrain = tempDataset.getUserNumRatings(i) / 2;
			int[] tempIndices = new int[tempNumItemsForTrain];
			for (int j = 0; j < tempNumItemsForTrain; j++) {
				tempIndices[j] = tempDataset.getTriple(i, j * 2).item;
			} // Of for j
			tempDataset.setUserTraining(i, tempIndices);

			// Step 3.2 Incremental training.
			tempLearner.trainUser(i);

			// Step 3.3 Prediction and compute error.
			int tempItem;
			double tempPrediction;
			for (int j = tempNumItemsForTrain; j < tempDataset.getUserNumRatings(i); j++) {
				tempItem = tempDataset.getTriple(i, j).item;
				tempPrediction = tempLearner.predict(i, tempItem);
				tempErrorSum += Math.abs(tempPrediction - tempDataset.getTriple(i, j).rating);
				tempNumPredictions++;
			} // Of for j

			// Step 3.4 Restore data of this user.
			tempDataset.setUserAllTraining(i);

			// Step 3.5 Show message.
			tempMAE = tempErrorSum / tempNumPredictions;
			System.out.println(
					"MAE = " + tempErrorSum + " / " + tempNumPredictions + " = " + tempMAE);
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
		testIncremental("data/jester-data-1/jester-data-1.txt", 24983, 101, 1810455, -10, 10, false,
				200, 100);
	}// Of main
}// Of class MF2DBooleanIncremental
