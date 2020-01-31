package algorithm;

import algorithm.MF2DBooleanIncremental;
import common.SimpleTools;
import gui.others.DoubleField;

import java.io.*;
import java.util.Arrays;
import java.util.Random;

/**
 * The main algorithm for three-way conversational recommendation. <br>
 * Project: Three-way conversational recommendation.<br>
 * 
 * @author Fan Min<br>
 *         www.fansmale.com, github.com/fansmale/TCR.<br>
 *         Email: minfan@swpu.edu.cn, minfanphd@163.com.<br>
 * @date Created: December 3, 2019.<br>
 *       Last modified: January 30, 2020.
 * @version 1.0
 */

public class TCR extends MF2DBooleanIncremental {

	/**
	 * The cost matrix.
	 */
	double[][] costMatrix;

	/**
	 * The maturity value array. For each recommendation to existing rating, the
	 * maturity increases by 5. For each recommendation to non-existing rating,
	 * the maturity increases by 2.5. For each promotion to existing rating, the
	 * maturity increases by 4. For each promotion to non-existing rating, the
	 * maturity increases by 2.
	 */
	double[] maturityValueArray = { 5, 2.5, 4, 2 };

	/**
	 * The default recommendation list length.
	 */
	public static final int DEFAULT_RECOMMENDATION_LENGTH = 10;

	/**
	 * The recommendation list length for each round.
	 */
	int recommendationLength;

	/**
	 * The default recommendation ratio.
	 */
	public static final double DEFAULT_RECOMMENDATION_RATIO = 0.3;

	/**
	 * Some items are recommended, while others are promoted.
	 */
	double recommendationRatio;

	/**
	 * Non-recommend.
	 */
	public static final int NON_RECOMMEND = 0;

	/**
	 * Promote.
	 */
	public static final int PROMOTE = 1;

	/**
	 * Recommend.
	 */
	public static final int RECOMMEND = 2;

	/**
	 * The default like threshold.
	 */
	public static final int DEFAULT_LIKE_THRESHOLD = 3;

	/**
	 * The like threshold.
	 */
	double likeThreshold;

	/**
	 * The default maturity threshold.
	 */
	public static final double DEFAULT_MATURITY_THRESHOLD = 1000;

	/**
	 * The maturity threshold.
	 */
	double maturityThreshold;

	/**
	 * The default maturity value for each item.
	 */
	public static final double DEFAULT_MATURITY_FOR_EACH_ITEM = 3;

	/**
	 * The maturity value for each item.
	 */
	double maturityForEachItem;

	/**
	 * The recommendation list for the current user.
	 */
	boolean[] currentUserRecommendations;

	/**
	 * The promotion list for the current user.
	 */
	boolean[] currentUserPromotions;

	/**
	 * The maximum of item pop.
	 */
	int maxItemPopularity;

	/**
	 * Item popularity no less than the second value may be recommended, no less
	 * than the first value may be promoted.
	 */
	double[] popularityThresholds = { 0.3, 0.5 };

	/**
	 * Item predicted rating no less than the second value may be recommended,
	 * no less than the first value may be promoted.
	 */
	double[] favoriteThresholds = { -2.0, 0.5 };

	/**
	 * The popular items (for recommend).
	 */
	int[] popularItems;

	/**
	 * The semi-popular items (for recommend).
	 */
	int[] semiPopopularItems;

	/**
	 * Neighborhood radius. For M-Distance based recommendation. double
	 * neighborhoodRadius;
	 */

	/**
	 * The statistics information (NN, NP, BN, ...) for current user.
	 */
	int[][] recommendationStatistics = new int[3][2];

	/**
	 *********************************** 
	 * The constructor.
	 * 
	 * @param paraFilename
	 *            The rating filename.
	 * @param paraNumUsers
	 *            The number of users.
	 * @param paraNumItems
	 *            The number of items.
	 * @param paraNumRatings
	 *            The number of ratings.
	 * @param paraRatingLowerBound
	 *            The lower bound of ratings.
	 * @param paraRatingUpperBound
	 *            The upper bound of ratings.
	 *********************************** 
	 */
	public TCR(String paraFilename, int paraNumUsers, int paraNumItems, int paraNumRatings,
			double paraRatingLowerBound, double paraRatingUpperBound) {
		// Step 1. Read data.
		super(paraFilename, paraNumUsers, paraNumItems, paraNumRatings, paraRatingLowerBound,
				paraRatingUpperBound);

		// Step 3. Initialize.
		initialize();
	}// Of the constructor

	/**
	 *********************************** 
	 * Initialize.
	 *********************************** 
	 */
	public void initialize() {
		recommendationLength = DEFAULT_RECOMMENDATION_LENGTH;
		recommendationRatio = DEFAULT_RECOMMENDATION_RATIO;

		costMatrix = new double[3][2];
		costMatrix[0][0] = 2; // NN
		costMatrix[0][1] = 40; // NP
		costMatrix[1][0] = 20; // BN
		costMatrix[1][1] = 10; // BP
		costMatrix[2][0] = 50; // PN
		costMatrix[2][1] = 6; // PP

		likeThreshold = DEFAULT_LIKE_THRESHOLD;
		maturityThreshold = DEFAULT_MATURITY_THRESHOLD;
		maturityForEachItem = DEFAULT_MATURITY_FOR_EACH_ITEM;

		maxItemPopularity = computeMaxItemPopularity();

		popularItems = null;
		semiPopopularItems = null;

		// neighborhoodRadius = 1.3;

		currentUserRecommendations = new boolean[numItems];
		currentUserPromotions = new boolean[numItems];
	}// Of initialize

	/**
	 *********************************** 
	 * Setter.
	 * 
	 * @param paraArray
	 *            The maturity value array with four values.
	 *********************************** 
	 */
	public void setMaturityValueArray(double[] paraArray) {
		maturityValueArray = paraArray;
	}// Of setMaturityValueArray

	/**
	 *********************************** 
	 * Setter.
	 *********************************** 
	 */
	public void setLikeThreshold(double paraValue) {
		likeThreshold = paraValue;
	}// Of setLikeThreshold

	/**
	 *********************************** 
	 * Getter.
	 *********************************** 
	 */
	public int[][] getRecommendationStatistics() {
		return recommendationStatistics;
	}// Of getRecommendationStatistics

	/**
	 *********************************** 
	 * Setter.
	 *********************************** 
	 */
	public void setMaturityThreshold(double paraValue) {
		maturityThreshold = paraValue;
	}// Of setMaturityThreshold

	/**
	 *********************************** 
	 * Setter.
	 * 
	 * @param paraThresholds
	 *            Should have exactly two elements.
	 *********************************** 
	 */
	public void setPopularityThresholds(double[] paraThresholds) {
		popularityThresholds = paraThresholds;
	}// Of setPopularityThresholds

	/**
	 *********************************** 
	 * Setter.
	 * 
	 * @param paraThresholds
	 *            Should have exactly two elements.
	 *********************************** 
	 */
	public void setFavoriteThresholds(double[] paraThreshold) {
		favoriteThresholds = paraThreshold;
	}// Of setFavoriteThresholds

	/**
	 *********************************** 
	 * Setter
	 *********************************** 
	 */
	public void setRecommendationLength(int paraLength) {
		recommendationLength = paraLength;
	}// Of setRecommendationLength

	/**
	 *********************************** 
	 * Setter
	 *********************************** 
	 */
	public void setRecommendationRatio(double paraRatio) {
		recommendationRatio = paraRatio;
	}// Of setRecommendationRatio

	/**
	 *************************** 
	 * Setter.
	 *************************** 
	 */
	public void setCostMatrix(double[][] paraCostMatrix) {
		costMatrix = paraCostMatrix;
	}// Of setCostMatrix

	/**
	 *********************************** 
	 * Setter.
	 *********************************** 
	 */
	public void setCostMatrix(int paraNN, int paraNP, int paraBN, int paraBP, int paraPN,
			int paraPP) {
		costMatrix[0][0] = paraNN;
		costMatrix[0][1] = paraNP;
		costMatrix[1][0] = paraBN;
		costMatrix[1][1] = paraBP;
		costMatrix[2][0] = paraPN;
		costMatrix[2][1] = paraPP;
		// Q.
	}// Of setCostMatrix

	/**
	 *********************************** 
	 * Compute both popular items, which may be recommended, and semi-popular
	 * items, which may be promoted.
	 * 
	 *********************************** 
	 */
	public void computePopAndSemipopItems(double paraPopThreshold, double paraSemiPopThreshold) {
		// Step 1. Compute popular items.
		int tempNumPopItems = 0;
		// Step 1.1 Compute the length
		for (int i = 0; i < itemPopularityArray.length; i++) {
			if (itemPopularityArray[i] > maxItemPopularity * paraPopThreshold) {
				tempNumPopItems++;
			} // Of if
		} // Of for i

		// Step 1.2 Compute the array
		popularItems = new int[tempNumPopItems];
		tempNumPopItems = 0;
		for (int i = 0; i < itemPopularityArray.length; i++) {
			if (itemPopularityArray[i] > maxItemPopularity * paraPopThreshold) {
				popularItems[tempNumPopItems] = i;
				tempNumPopItems++;
			} // Of if
		} // Of for i

		// Step 2. Compute semi-popular items.
		int tempNumSemiPopItems = 0;
		// Step 2.1 Compute the length
		for (int i = 0; i < itemPopularityArray.length; i++) {
			if ((itemPopularityArray[i] > maxItemPopularity * paraSemiPopThreshold)
					&& (itemPopularityArray[i] <= maxItemPopularity * paraPopThreshold)) {
				tempNumSemiPopItems++;
			} // Of if
		} // Of for i

		// Step 1.2 Compute the array
		semiPopopularItems = new int[tempNumSemiPopItems];
		tempNumSemiPopItems = 0;
		for (int i = 0; i < itemPopularityArray.length; i++) {
			if ((itemPopularityArray[i] > maxItemPopularity * paraSemiPopThreshold)
					&& (itemPopularityArray[i] <= maxItemPopularity * paraPopThreshold)) {
				semiPopopularItems[tempNumSemiPopItems] = i;
				tempNumSemiPopItems++;
			} // Of if
		} // Of for i

		// System.out.println("Pop items: " + Arrays.toString(popularItems));
		// System.out.println("Semi-popular items: " +
		// Arrays.toString(semiPopopularItems));
	}// Of computePopAndSemipopItems

	/**
	 *********************************** 
	 * Compute the maximum of item pop.
	 *********************************** 
	 */
	public int computeMaxItemPopularity() {
		int resultMaxPop = itemPopularityArray[0];
		for (int i = 1; i < itemPopularityArray.length; i++) {
			if (resultMaxPop < itemPopularityArray[i]) {
				resultMaxPop = itemPopularityArray[i];
			} // Of if
		} // Of for i
		return resultMaxPop;
	}// Of computeMaxItemPopularity

	/**
	 *********************************** 
	 * Compute the total cost for a user.
	 * 
	 * @param paraUser
	 *            The index of the given user.
	 * @param paraRecommendations
	 *            The recommendations to the user, true for recommendation.
	 * @param paraPromotions
	 *            The promotions for the user, true for promotion.
	 * @return The total cost for the user.
	 *********************************** 
	 */
	public double computeTotalCostForUser(int paraUser) {
		return computeTotalCostForUser(paraUser, currentUserRecommendations, currentUserPromotions);
	}// Of computeTotalCostForUser

	/**
	 *********************************** 
	 * Compute the total cost for a user.
	 * 
	 * @param paraUser
	 *            The index of the given user.
	 * @param paraRecommendations
	 *            The recommendations to the user, true for recommendation.
	 * @param paraPromotions
	 *            The promotions for the user, true for promotion.
	 * @return The total cost for the user.
	 *********************************** 
	 */
	public double computeTotalCostForUser(int paraUser, boolean[] paraRecommendations,
			boolean[] paraPromotions) {

		// Step 1. Check them.
		double resultTotalCost = 0;
		int tempBehavior;
		int tempLike;
		for (int i = 0; i < data[paraUser].length; i++) {
			if (paraRecommendations[data[paraUser][i].item]) {
				tempBehavior = RECOMMEND;
			} else if (paraPromotions[data[paraUser][i].item]) {
				tempBehavior = PROMOTE;
			} else {
				tempBehavior = NON_RECOMMEND;
			} // Of if

			tempLike = 0;// 0 means "dislike".
			if (data[paraUser][i].rating > likeThreshold) {
				tempLike = 1;// 1 means "like".
			} // Of if

			resultTotalCost += costMatrix[tempBehavior][tempLike];
		} // Of for i

		return resultTotalCost;
	}// Of computeTotalCostForUser

	/**
	 *********************************** 
	 * Compute recommendation statistics for a user.
	 * 
	 * @param paraUser
	 *            The index of the given user.
	 * @return The recommendation statistics for the user.
	 *********************************** 
	 */
	public int[][] computeUserRecommendationStatitics(int paraUser) {
		int[][] resultUserRecommendationStatistics = new int[3][2];

		// Step 1. Check them.
		int tempBehavior;
		int tempLike;
		for (int i = 0; i < data[paraUser].length; i++) {
			if (currentUserRecommendations[data[paraUser][i].item]) {
				tempBehavior = RECOMMEND;
			} else if (currentUserPromotions[data[paraUser][i].item]) {
				tempBehavior = PROMOTE;
			} else {
				tempBehavior = NON_RECOMMEND;
			} // Of if

			tempLike = 0;// 0 means "dislike".
			if (data[paraUser][i].rating > likeThreshold) {
				tempLike = 1;// 1 means "like".
			} // Of if

			resultUserRecommendationStatistics[tempBehavior][tempLike]++;
		} // Of for i

		return resultUserRecommendationStatistics;
	}// Of computeUserRecommendationStatitics

	/**
	 *********************************** 
	 * Leave-user-out recommendation.
	 * 
	 * @return The total cost of the current user.
	 *********************************** 
	 */
	public double leaveUserOutRecommend() {
		double resultTotalCost = 0;
		recommendationStatistics = new int[3][2];
		System.out.println("leaveUserOutRecommend(), numUsers = " + numUsers);
		for (int i = 0; i < numUsers; i++) {
			if (i % 100 == 0) {
				SimpleTools.processTrackingOutput("Recommending for user #" + i + ":");
			} // Of if
			recommendForUser(i);
			resultTotalCost += recommendForUser(i);
			int[][] tempUserStatistics = computeUserRecommendationStatitics(i);
			for (int j = 0; j < tempUserStatistics.length; j++) {
				for (int k = 0; k < tempUserStatistics[0].length; k++) {
					recommendationStatistics[j][k] += tempUserStatistics[j][k];
				} // Of for k
			} // Of for j
		} // Of for i

		return resultTotalCost;
	}// Of leaveUserOutRecommend

	/**
	 *********************************** 
	 * Recommend for one user.
	 * 
	 * @param paraUser
	 *            The user index.
	 * @return The total cost of the current user.
	 *********************************** 
	 */
	public double recommendForUser(int paraUser) {
		SimpleTools.processTrackingOutput("\r\nUser " + paraUser);
		// Step 1. Initialize
		double resultTotalCost;
		Arrays.fill(currentUserRecommendations, false);
		Arrays.fill(currentUserPromotions, false);

		// Step 2. Popularity-based recommendation.
		popularityBasedRecommend(paraUser);
		resultTotalCost = computeTotalCostForUser(paraUser, currentUserRecommendations,
				currentUserPromotions);
		SimpleTools.variableTrackingOutput("User " + paraUser
				+ ", after popularity based recommendation, total cost = " + resultTotalCost);
		// System.out.print("Only popularity-based recommendtaion. The cost of
		// user " + paraUser
		// + " is: " + resultTotalCost);
		// System.out.println();

		// Step 3. M-distance-based recommendation.
		// mDistanceBasedRecommend(paraUser);

		// Step 3. GLMF recommendation.
		// MF2DBooleanIncremental(paraUser);

		// Step 3. MF based recommendation.
		mfBasedRecommend(paraUser);

		resultTotalCost = computeTotalCostForUser(paraUser, currentUserRecommendations,
				currentUserPromotions);
		SimpleTools.variableTrackingOutput("User " + paraUser
				+ ", after MF based recommendation, total cost = " + resultTotalCost);
		// System.out.print("Finally, the cost of user " + paraUser + " is: " +
		// resultTotalCost);
		// System.out.println();

		return resultTotalCost;
	}// Of recommendForUser

	/**
	 *********************************** 
	 * Pop-based recommendation. With member variables
	 * currentUserRecommendations and currentUserPromotions, there is no need to
	 * return.
	 * 
	 * @param paraUser
	 *            The given user.
	 *********************************** 
	 */
	public void popularityBasedRecommend(int paraUser) {
		// Step 1. Initialize the total/average score of each item.
		int[] tempItemPopArray = new int[numItems];
		for (int i = 0; i < numItems; i++) {
			tempItemPopArray[i] = itemPopularityArray[i];
		} // Of for i

		// Remove those for the current user since the data is unknown.
		for (int i = 0; i < data[paraUser].length; i++) {
			tempItemPopArray[data[paraUser][i].item]--;
		} // Of for i

		// Step 2. Compute popular items.
		double tempMaturity = 0;
		int[] tempPopItems = new int[numItems];
		int tempNumPopItems = 0;
		// System.out.print("Popular items: ");
		for (int i = 0; i < numItems; i++) {
			if (tempItemPopArray[i] >= popularityThresholds[1] * maxItemPopularity) {
				tempPopItems[tempNumPopItems] = i;
				// System.out.print(", " + i);
				tempNumPopItems++;
			} // Of for i
		} // Of for i

		// Step 3. Compute semi-popular items.
		int[] tempSemiPopItems = new int[numItems];
		int tempNumSemiPopItems = 0;
		// System.out.print("\r\nSemi-popular items: ");
		for (int i = 0; i < numItems; i++) {
			if ((tempItemPopArray[i] < popularityThresholds[1] * maxItemPopularity)
					&& (tempItemPopArray[i] >= popularityThresholds[0] * maxItemPopularity)) {
				tempSemiPopItems[tempNumSemiPopItems] = i;
				// System.out.print("; " + i);
				tempNumSemiPopItems++;
			} // Of for i
		} // Of for i

		// Step 4. Pop-based recommendation.
		int tempNumRecommend = (int) (recommendationRatio * recommendationLength);
		int tempNumPromote = recommendationLength - tempNumRecommend;
		boolean[] tempProcessedArray = new boolean[numItems];
		Arrays.fill(tempProcessedArray, false);

		while (tempMaturity < maturityThreshold) {
			if (tempNumRecommend > tempNumPopItems) {
				SimpleTools.processTrackingOutput("No enough to recommend.");
				break;
			} // Of if

			// Step 4.1 Randomly select some to recommend
			int[] tempRecommendations = randomSelectFromArray(tempPopItems, tempNumPopItems,
					tempNumRecommend);
			for (int i = 0; i < tempRecommendations.length; i++) {
				tempProcessedArray[tempRecommendations[i]] = true;
				currentUserRecommendations[tempRecommendations[i]] = true;
			} // Of for i
			tempNumPopItems -= tempNumRecommend;
			tempPopItems = eliminateProcessed(tempPopItems, tempProcessedArray);
			// System.out.println("\r\nRecommending " +
			// Arrays.toString(tempRecommendations));

			// Step 4.2 Promote
			// Step 4.2.1 Compute items with popularity between the pop and
			// promotion thresholds.
			if (tempNumPromote > tempNumSemiPopItems) {
				SimpleTools.processTrackingOutput("No enough to promote.");
				break;
			} // Of if

			// Step 4.2.2 Randomly select 7 to promote
			int[] tempPromotions = randomSelectFromArray(tempSemiPopItems, tempNumSemiPopItems,
					tempNumPromote);
			// System.out.println("\r\nPromoting " +
			// Arrays.toString(tempPromotions));
			tempNumSemiPopItems -= tempNumPromote;
			for (int i = 0; i < tempPromotions.length; i++) {
				tempProcessedArray[tempPromotions[i]] = true;
				currentUserPromotions[tempPromotions[i]] = true;
			} // Of for i
			tempSemiPopItems = eliminateProcessed(tempSemiPopItems, tempProcessedArray);

			// Update current user recommendations
			for (int i = 0; i < tempRecommendations.length; i++) {
				currentUserRecommendations[tempRecommendations[i]] = true;
			} // Of for i
				// Update current user promotions
			for (int i = 0; i < tempPromotions.length; i++) {
				currentUserPromotions[tempPromotions[i]] = true;
			} // Of for i

			// Step 3 Update the maturity
			tempMaturity += computeUserMaturity(paraUser, currentUserRecommendations,
					currentUserPromotions);
		} // Of while

		if (tempMaturity >= maturityThreshold) {
			SimpleTools.processTrackingOutput("Matured.");
		} else {
			SimpleTools.processTrackingOutput("The maturity " + tempMaturity
					+ " is smaller than the threshold " + maturityThreshold);
		} // Of if

	}// Of popularityBasedRecommend

	/**
	 *********************************** 
	 * MF-based recommendation. With member variables currentUserRecommendations
	 * and currentUserPromotions, there is no need to return.
	 * 
	 * @param paraUser
	 *            The given user.
	 *********************************** 
	 */
	public void mfBasedRecommend(int paraUser) {
		// System.out.println("mfBasedRecommend");
		int[] tempAcquiredItems = new int[data[paraUser].length];
		int tempCounter = 0;
		int[] tempRecommendationCandidates = new int[numItems];
		int[] tempPromotionCandidates = new int[numItems];

		int tempNumRecommend = (int) (recommendationRatio * recommendationLength);
		int tempNumPromote = recommendationLength - tempNumRecommend;

		int[] tempRandomIndices = null;

		while (true) {
			// Step 1. Which items have rating information available.
			tempCounter = 0;
			for (int i = 0; i < data[paraUser].length; i++) {
				if (currentUserRecommendations[data[paraUser][i].item]
						|| currentUserPromotions[data[paraUser][i].item]) {
					tempAcquiredItems[tempCounter] = data[paraUser][i].item;
					tempCounter++;
				} // Of if
			} // Of for i

			// Compress
			int[] tempCompressedItems = new int[tempCounter];
			for (int i = 0; i < tempCounter; i++) {
				tempCompressedItems[i] = tempAcquiredItems[i];
			} // Of for i

			// Step 2. Predict for the current user.
			setUserTraining(paraUser, tempCompressedItems);
			trainUser(paraUser);
			double[] tempPredicts = predictForUser(paraUser);

			// Step 3. Generate recommendation/promotion candidates list
			int tempRecommendationCandidatesLength = 0;
			int tempPromotionCandidatesLength = 0;

			for (int i = 0; i < numItems; i++) {
				// Already recommended/promoted before
				if (currentUserRecommendations[i] || currentUserPromotions[i]) {
					continue;
				} // Of if

				// System.out.println("tempPredicts[" + i + "]= " +
				// tempPredicts[i]);
				if (tempPredicts[i] >= favoriteThresholds[1]) {
					tempRecommendationCandidates[tempRecommendationCandidatesLength] = i;
					tempRecommendationCandidatesLength++;
				} else if (tempPredicts[i] >= favoriteThresholds[0]) {
					tempPromotionCandidates[tempPromotionCandidatesLength] = i;
					tempPromotionCandidatesLength++;
				} // Of if
			} // Of for i

			// Step 4. Handle the situation where no enough to
			// recommend/promote.
			if (tempRecommendationCandidatesLength < tempNumRecommend) {
				// System.out.println("User " + paraUser + " has no enough to
				// recommend: "
				// + tempRecommendationCandidatesLength);
				break;
			} else if (tempPromotionCandidatesLength < tempNumPromote) {
				// System.out.println("User " + paraUser + " has no enough to
				// promote: "
				// + tempPromotionCandidatesLength);
				break;
			} // Of if

			// Step 5. Randomly select some to recommend/promote.
			try {
				tempRandomIndices = SimpleTools.generateRandomIndices(
						tempRecommendationCandidatesLength, tempNumRecommend);
				for (int j = 0; j < tempNumRecommend; j++) {
					currentUserRecommendations[tempRecommendationCandidates[tempRandomIndices[j]]] = true;
					// System.out.println("Recommend: " +
					// tempRecommendationCandidates[tempRandomIndices[j]]);
				} // Of for j

				tempRandomIndices = SimpleTools.generateRandomIndices(tempPromotionCandidatesLength,
						tempNumPromote);
				for (int j = 0; j < tempNumPromote; j++) {
					currentUserPromotions[tempPromotionCandidates[tempRandomIndices[j]]] = true;
					// System.out.println("Promote: " +
					// tempPromotionCandidates[tempRandomIndices[j]]);
				} // Of for j
			} catch (Exception ee) {
				System.out.println("Error occurred in TIR.mfBasedRecommend(int)\r\n" + ee);
			} // Of try
		} // Of while
	}// Of mfBasedRecommend

	/**
	 *********************************** 
	 * Randomly select some elements from the given array.
	 * 
	 * @param paraArray
	 *            The given array.
	 * @param paraValidLength
	 *            Valid length of the array.
	 * @param paraNumSelection
	 *            The number of selected elements.
	 *********************************** 
	 */
	public static int[] randomSelectFromArray(int[] paraArray, int paraValidLength,
			int paraNumSelection) {

		int[] tempArray = null;
		try {
			tempArray = SimpleTools.generateRandomIndices(paraValidLength, paraNumSelection);
		} catch (Exception ee) {
			System.out.println("Internal error occurred in TCR.randomSelectFromArray(): \r\n" + ee);
			System.exit(0);
		} // Of try

		int[] resultArray = new int[paraNumSelection];
		for (int i = 0; i < resultArray.length; i++) {
			resultArray[i] = paraArray[tempArray[i]];
		} // Of for i

		return resultArray;
	}// Of randomSelectFromArray

	/**
	 *********************************** 
	 * Eliminate elements that are already processed.
	 * 
	 * @param paraArray
	 *            The given array.
	 * @param paraProcessed
	 *            Indicate which elements have been processed.
	 * @return The new array.
	 *********************************** 
	 */
	public static int[] eliminateProcessed(int[] paraArray, boolean[] paraProcessed) {
		int[] tempArray = new int[paraArray.length];
		int tempLength = 0;

		// Step 1. Copy unprocessed elements.
		for (int i = 0; i < paraArray.length; i++) {
			if (!paraProcessed[paraArray[i]]) {
				tempArray[tempLength] = paraArray[i];
				tempLength++;
			} // Of if
		} // Of if

		// Step 2. Compress.
		int[] resultArray = new int[tempLength];
		for (int i = 0; i < tempLength; i++) {
			resultArray[i] = tempArray[i];
		} // Of for i
		return resultArray;
	}// Of eliminateProcessed

	/**
	 *********************************** 
	 * Compute the user maturity increment. For each recommendation to existing
	 * rating, the maturity increases by 5. For each recommendation to
	 * non-existing rating, the maturity increases by 2.5. For each promotion to
	 * existing rating, the maturity increases by 4. For each promotion to
	 * non-existing rating, the maturity increases by 2.
	 * 
	 * @param paraUser
	 *            The given user.
	 * @param paraRecommendations
	 *            The recommendation array.
	 * @param paraPromotions
	 *            The promotion array.
	 *********************************** 
	 */
	public double computeUserMaturity(int paraUser, boolean[] paraRecommendations,
			boolean[] paraPromotions) {
		double resultValue = 0;
		// Step 1. Which items are actually rated.
		boolean[] tempUserBehaviors = new boolean[numItems];
		for (int i = 0; i < data[paraUser].length; i++) {
			tempUserBehaviors[data[paraUser][i].item] = true;
		} // Of for i

		// 0 for no recommendation/promotion
		for (int i = 0; i < numItems; i++) {
			if (paraRecommendations[i]) {
				if (tempUserBehaviors[i]) {
					resultValue += maturityValueArray[0];
				} else {
					resultValue += maturityValueArray[1];
				} // Of if
			} else if (paraPromotions[i]) {
				if (tempUserBehaviors[i]) {
					resultValue += maturityValueArray[2];
				} else {
					resultValue += maturityValueArray[3];
				} // Of if
			} // Of for i
		} // Of for i

		return resultValue;
	}// Of computeUserMaturity

	/**
	 *********************************** 
	 * The main entrance.
	 * 
	 * @throws IOException
	 * @throws NumberFormatException
	 *********************************** 
	 */
	public static void main(String args[]) {
		// TIR2 tir = new TIR2("data/movielens100k.data", 943, 1682, 100000,
		// -10, 10);
		TCR tcr = new TCR("data/jester-data-1/jester-data-1.txt", 24983, 101, 1810455, -10, 10);
		System.out.println(tcr);

		tcr.setPopularityThresholds(new double[] { 0.5, 0.9 });

		double tempTotalCost = tcr.leaveUserOutRecommend();

		System.out.println("The total cost for all users is: " + tempTotalCost);
	}// Of main

}// Of class TIR
