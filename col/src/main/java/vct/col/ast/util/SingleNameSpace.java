package vct.col.ast.util;

import hre.lang.HREException;
import hre.lang.HREExitException;
import vct.col.ast.expr.NameExpressionKind;

import java.util.*;

import static hre.lang.System.DebugException;
import static hre.lang.System.Fail;

/**
 * A name space with a single definition per name.
 * 
 * @author Stefan Blom
 *
 */
public class SingleNameSpace implements Map<String, VariableInfo> {

  protected Stack<Map<String, VariableInfo>> stack=new Stack<Map<String, VariableInfo>>();
  
  protected Map<String, VariableInfo> map=new HashMap<String, VariableInfo>();
 
  public void enter(){
    stack.push(map);
    map=new HashMap<String, VariableInfo>();
    map.putAll(stack.peek());
  }
  public VariableInfo lookup(String name){
    return map.get(name);
  }
  public void add(String name, VariableInfo def){
    if(map.containsKey(name)){
      var original = map.get(name);
      if(original.kind == NameExpressionKind.Field && def.kind == NameExpressionKind.Argument){
        def.reference.getOrigin().report("","Duplicate declaration found");
        map.get(name).reference.getOrigin().report("","Original declaration");
        Fail("");
      }
    }
    map.put(name, def);
  }
  public void leave(){
    map=stack.pop();
  }
  
  @Override
  public int size() {
    return map.size();
  }
  @Override
  public boolean isEmpty() {
    return map.isEmpty();
  }
  @Override
  public boolean containsKey(Object key) {
     return map.containsKey(key);
  }
  @Override
  public boolean containsValue(Object value) {
    return map.containsValue(value);
  }
  @Override
  public VariableInfo get(Object key) {
    return map.get(key);
  }
  @Override
  public VariableInfo put(String key, VariableInfo value) {
    return map.put(key, value);
  }
  @Override
  public VariableInfo remove(Object key) {
    return map.remove(key);
  }
  @Override
  public void putAll(Map<? extends String, ? extends VariableInfo> m) {
    map.putAll(m);
  }
  @Override
  public void clear() {
    map.clear();
  }
  @Override
  public Set<String> keySet() {
    return map.keySet();
  }
  @Override
  public Collection<VariableInfo> values() {
    return map.values();
  }
  @Override
  public Set<java.util.Map.Entry<String, VariableInfo>> entrySet() {
    return map.entrySet();
  }
}
