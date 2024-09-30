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

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.lang.invoke.MethodHandles;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.tolstoy.aiaiai.api.IConfidenceMatrix;

class ConfidenceMatrix implements IConfidenceMatrix {
	private static final Logger logger = LogManager.getLogger( MethodHandles.lookup().lookupClass() );

	private final Map<String,Integer> classCounts, matrix;

	ConfidenceMatrix( Map<String,Integer> classCounts ) {
		this.classCounts = classCounts;
		this.matrix = new HashMap<String,Integer>();

		for ( String expected : classCounts.keySet() ) {
			for ( String actual : classCounts.keySet() ) {
				this.matrix.put( createKey( expected, actual ), 0 );
			}
		}
	}

	public void addPrediction( String expected, String actual ) {
		String key = createKey( expected, actual );

		matrix.put( key, matrix.get( key ) + 1 );
	}

	public List<String> getResults() {
		List<String> ret = new ArrayList<String>();

		for ( String expected : classCounts.keySet() ) {
			int numCorrect = matrix.get( createKey( expected, expected ) );
			String msg = "For " + expected + ", num correct=" + numCorrect + ".";
			for ( String actual : classCounts.keySet() ) {
				if ( expected.equals( actual ) ) {
					continue;
				}
				msg += " Num incorrectly coded as " + actual + "=" + matrix.get( createKey( expected, actual ) ) + ".";
			}

			ret.add( msg );
		}

		return ret;
	}

	public Map<String,Integer> getClassCounts() {
		return new HashMap<String,Integer>( classCounts );
	}

	public Map<String,Integer> getMatrix() {
		return new HashMap<String,Integer>( matrix );
	}

	protected String createKey( String expected, String actual ) {
		return expected + ",,,,," + actual;
	}
}
