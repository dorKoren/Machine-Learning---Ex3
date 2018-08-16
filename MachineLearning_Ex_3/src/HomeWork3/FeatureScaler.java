package HomeWork3;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class FeatureScaler {
	/**
	 * Returns a scaled version (using standarized normalization) of the given dataset.
	 * @param instances The original dataset.
	 * @return A scaled instances object.
	 */
	public static Instances scaleData(Instances instances) throws Exception{    // הוספתי סטטי!!!!!!!!
		Standardize standardization = new Standardize();
		standardization.setInputFormat(instances);
		Instances scaledInstances = Filter.useFilter(instances, standardization);
	
		return scaledInstances;
	}
}