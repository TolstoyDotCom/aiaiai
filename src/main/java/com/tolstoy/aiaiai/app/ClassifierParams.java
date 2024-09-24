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

import weka.core.Attribute;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.tolstoy.aiaiai.api.IClassifierParams;
import com.tolstoy.aiaiai.api.AttributeNotSetException;

class ClassifierParams<V> implements IClassifierParams<V> {
	private static final Logger logger = LogManager.getLogger( MethodHandles.lookup().lookupClass() );

	private final List<Attribute> valueAttributes;
	private final Attribute classAttribute;
	private final Map<String,V> map;
	private final int maxValueAttributeIndex;

	ClassifierParams( Attribute classAttribute, List<Attribute> valueAttributes ) {
		this.classAttribute = classAttribute;
		this.valueAttributes = valueAttributes;
		this.map = new HashMap<String,V>();

		int max = -1;
		for ( Attribute attr : valueAttributes ) {
			max = Math.max( max, attr.index() );
		}

		this.maxValueAttributeIndex = max;
	}

	public List<V> getList() throws AttributeNotSetException {
		List<V> ret = new ArrayList<V>( maxValueAttributeIndex + 1 );
		for ( int i = 0; i < maxValueAttributeIndex + 1; i++ ) {
			ret.add( null );
		}
		for ( Attribute attr : valueAttributes ) {
			V val = map.get( attr.name() );
			if ( val == null ) {
				throw new AttributeNotSetException( attr.name() );
			}

			ret.set( attr.index(), val );
		}

		return ret;
	}

	public V getValue( String key ) {
		Attribute attr = findValueAttribute( key );
		if ( attr == null ) {
			throw new IllegalArgumentException( "Key is not valid: " + key );
		}

		return map.get( key );
	}

	public void setValue( String key, V value ) {
		Attribute attr = findValueAttribute( key );
		if ( attr == null ) {
			throw new IllegalArgumentException( "Key is not valid: " + key );
		}

		map.put( key, value );
	}

	public void clear() {
		map.clear();
	}

	protected Attribute findValueAttribute( String key ) {
		for ( Attribute attr : valueAttributes ) {
			if ( attr.name().equals( key ) ) {
				return attr;
			}
		}

		return null;
	}

	@Override
	public String toString() {
		return "class=" + classAttribute + ", values=" + valueAttributes + ", map=" + map + ", maxValueAttributeIndex=" + maxValueAttributeIndex;
	}
}
