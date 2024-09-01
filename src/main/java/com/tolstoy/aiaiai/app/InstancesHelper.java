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
import java.util.Enumeration;
import java.lang.invoke.MethodHandles;

import weka.core.Attribute;
import weka.core.Instances;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

class InstancesHelper {
	private static final Logger logger = LogManager.getLogger( MethodHandles.lookup().lookupClass() );

	private final Instances base;

	InstancesHelper( Instances base ) {
		this.base = new Instances( base );

		Attribute tempClassAttribute = null;

		Enumeration<Attribute> e = this.base.enumerateAttributes();
		while ( e.hasMoreElements() ) {
			Attribute a = e.nextElement();
			if ( a.enumerateValues() != null ) {
				if ( tempClassAttribute != null ) {
					throw new IllegalArgumentException( "Found multiple class attributes" );
				}

				tempClassAttribute = a;
			}
		}
	}

	Instances getInstances() {
		return base;
	}

	List<Attribute> getValueAttributes() {
		List<Attribute> ret = new ArrayList<Attribute>();

		Enumeration<Attribute> e = base.enumerateAttributes();
		while ( e.hasMoreElements() ) {
			Attribute a = e.nextElement();
			if ( a.enumerateValues() == null ) {
				ret.add( a );
			}
		}

		return ret;
	}

	Attribute getClassAttribute() {
		Enumeration<Attribute> e = this.base.enumerateAttributes();
		while ( e.hasMoreElements() ) {
			Attribute a = e.nextElement();
			if ( a.enumerateValues() != null ) {
				return a;
			}
		}

		throw new RuntimeException( "Class attribute not found" );
	}

	List<String> getClassAttributeOptions() {
		Attribute attr = getClassAttribute();

		Enumeration<Object> e = attr.enumerateValues();
		if ( e == null ) {
			throw new RuntimeException( "Class attribute has no options" );
		}

		List<String> ret = new ArrayList<String>();

		while ( e.hasMoreElements() ) {
			Object obj = e.nextElement();
			if ( obj instanceof String ) {
				ret.add( (String) obj );
			}
		}

		return ret;
	}
}
