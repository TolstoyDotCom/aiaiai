/*
 * Copyright 2024 Chris Kelly
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

package com.tolstoy.aiaiai.app;

import java.nio.file.Paths;
import java.io.File;
import java.util.List;
import java.util.ArrayList;
import java.util.Set;
import java.lang.invoke.MethodHandles;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import weka.core.Instances;

import com.tolstoy.aiaiai.api.IClassifierBuilderFactory;
import com.tolstoy.aiaiai.api.IClassifierBuilder;
import com.tolstoy.aiaiai.api.IClassifierParams;
import com.tolstoy.aiaiai.api.IKeyedInstanceSet;
import com.tolstoy.aiaiai.api.IKeyedInstance;

public class App {
	private static final Logger logger = LogManager.getLogger( MethodHandles.lookup().lookupClass() );

	private static final String CLASSIFIER_NAME = "weka.classifiers.trees.J48";
	private static final String FILTER_NAME = "weka.filters.unsupervised.instance.Randomize";

	public static void main( String[] args ) {
		try {
			List<String> classifierArguments = new ArrayList<String>();
			classifierArguments.add( "-U" );

			File inputFile = Paths.get( App.class.getResource( "/iris.arff" ).toURI() ).toFile();

			IClassifierBuilderFactory classifierBuilderFactory = new ClassifierBuilderFactory();

			IClassifierBuilder builder = classifierBuilderFactory.createClassifierBuilder( CLASSIFIER_NAME, classifierArguments, FILTER_NAME, null, inputFile );

			IClassifierParams params = builder.createClassifierParams();

			IKeyedInstanceSet keyedInstanceSet = new KeyedInstanceSet<String,Double>( inputFile );

			int numMatches = 0;

			List<IKeyedInstance<String,Double>> keyedInstances = keyedInstanceSet.getKeyedInstances();
			for ( IKeyedInstance<String,Double> keyedInstance : keyedInstances ) {
				String expected = keyedInstance.getExpectedClass();
				Set<String> keys = keyedInstance.getKeys();
				for ( String key : keys ) {
					params.setValue( key, keyedInstance.getValue( key ) );
				}

				String actual = builder.classify( params );

				if ( expected.equals( actual ) ) {
					numMatches++;
				}
			}

			double pct = ( 100 * numMatches ) / keyedInstances.size();

			logger.info( "Out of " + keyedInstances.size() + " instances, " + numMatches + " matched the value in the .arff file, for a success percent of " + pct );
		}
		catch ( Exception e ) {
			logger.catching( e );
		}
	}
}
