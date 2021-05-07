package io.deephaven.javascript.proto.dhinternal.io.deephaven.proto.table_pb;

import elemental2.core.JsArray;
import elemental2.core.Uint8Array;
import io.deephaven.javascript.proto.dhinternal.io.deephaven.proto.session_pb.Ticket;
import jsinterop.annotations.JsOverlay;
import jsinterop.annotations.JsPackage;
import jsinterop.annotations.JsProperty;
import jsinterop.annotations.JsType;
import jsinterop.base.Js;
import jsinterop.base.JsPropertyMap;

@JsType(
    isNative = true,
    name = "dhinternal.io.deephaven.proto.table_pb.SortTableRequest",
    namespace = JsPackage.GLOBAL)
public class SortTableRequest {
  @JsType(isNative = true, name = "?", namespace = JsPackage.GLOBAL)
  public interface ToObjectReturnType {
    @JsType(isNative = true, name = "?", namespace = JsPackage.GLOBAL)
    public interface ResultidFieldType {
      @JsType(isNative = true, name = "?", namespace = JsPackage.GLOBAL)
      public interface GetIdUnionType {
        @JsOverlay
        static SortTableRequest.ToObjectReturnType.ResultidFieldType.GetIdUnionType of(Object o) {
          return Js.cast(o);
        }

        @JsOverlay
        default String asString() {
          return Js.asString(this);
        }

        @JsOverlay
        default Uint8Array asUint8Array() {
          return Js.cast(this);
        }

        @JsOverlay
        default boolean isString() {
          return (Object) this instanceof String;
        }

        @JsOverlay
        default boolean isUint8Array() {
          return (Object) this instanceof Uint8Array;
        }
      }

      @JsOverlay
      static SortTableRequest.ToObjectReturnType.ResultidFieldType create() {
        return Js.uncheckedCast(JsPropertyMap.of());
      }

      @JsProperty
      SortTableRequest.ToObjectReturnType.ResultidFieldType.GetIdUnionType getId();

      @JsProperty
      void setId(SortTableRequest.ToObjectReturnType.ResultidFieldType.GetIdUnionType id);

      @JsOverlay
      default void setId(String id) {
        setId(
            Js.<SortTableRequest.ToObjectReturnType.ResultidFieldType.GetIdUnionType>uncheckedCast(
                id));
      }

      @JsOverlay
      default void setId(Uint8Array id) {
        setId(
            Js.<SortTableRequest.ToObjectReturnType.ResultidFieldType.GetIdUnionType>uncheckedCast(
                id));
      }
    }

    @JsType(isNative = true, name = "?", namespace = JsPackage.GLOBAL)
    public interface SortsListFieldType {
      @JsOverlay
      static SortTableRequest.ToObjectReturnType.SortsListFieldType create() {
        return Js.uncheckedCast(JsPropertyMap.of());
      }

      @JsProperty
      String getColumnname();

      @JsProperty
      double getDirection();

      @JsProperty
      boolean isIsabsolute();

      @JsProperty
      void setColumnname(String columnname);

      @JsProperty
      void setDirection(double direction);

      @JsProperty
      void setIsabsolute(boolean isabsolute);
    }

    @JsType(isNative = true, name = "?", namespace = JsPackage.GLOBAL)
    public interface SourceidFieldType {
      @JsOverlay
      static SortTableRequest.ToObjectReturnType.SourceidFieldType create() {
        return Js.uncheckedCast(JsPropertyMap.of());
      }

      @JsProperty
      double getBatchoffset();

      @JsProperty
      Object getTicket();

      @JsProperty
      void setBatchoffset(double batchoffset);

      @JsProperty
      void setTicket(Object ticket);
    }

    @JsOverlay
    static SortTableRequest.ToObjectReturnType create() {
      return Js.uncheckedCast(JsPropertyMap.of());
    }

    @JsProperty
    SortTableRequest.ToObjectReturnType.ResultidFieldType getResultid();

    @JsProperty
    JsArray<SortTableRequest.ToObjectReturnType.SortsListFieldType> getSortsList();

    @JsProperty
    SortTableRequest.ToObjectReturnType.SourceidFieldType getSourceid();

    @JsProperty
    void setResultid(SortTableRequest.ToObjectReturnType.ResultidFieldType resultid);

    @JsProperty
    void setSortsList(JsArray<SortTableRequest.ToObjectReturnType.SortsListFieldType> sortsList);

    @JsOverlay
    default void setSortsList(SortTableRequest.ToObjectReturnType.SortsListFieldType[] sortsList) {
      setSortsList(
          Js.<JsArray<SortTableRequest.ToObjectReturnType.SortsListFieldType>>uncheckedCast(
              sortsList));
    }

    @JsProperty
    void setSourceid(SortTableRequest.ToObjectReturnType.SourceidFieldType sourceid);
  }

  @JsType(isNative = true, name = "?", namespace = JsPackage.GLOBAL)
  public interface ToObjectReturnType0 {
    @JsType(isNative = true, name = "?", namespace = JsPackage.GLOBAL)
    public interface ResultidFieldType {
      @JsType(isNative = true, name = "?", namespace = JsPackage.GLOBAL)
      public interface GetIdUnionType {
        @JsOverlay
        static SortTableRequest.ToObjectReturnType0.ResultidFieldType.GetIdUnionType of(Object o) {
          return Js.cast(o);
        }

        @JsOverlay
        default String asString() {
          return Js.asString(this);
        }

        @JsOverlay
        default Uint8Array asUint8Array() {
          return Js.cast(this);
        }

        @JsOverlay
        default boolean isString() {
          return (Object) this instanceof String;
        }

        @JsOverlay
        default boolean isUint8Array() {
          return (Object) this instanceof Uint8Array;
        }
      }

      @JsOverlay
      static SortTableRequest.ToObjectReturnType0.ResultidFieldType create() {
        return Js.uncheckedCast(JsPropertyMap.of());
      }

      @JsProperty
      SortTableRequest.ToObjectReturnType0.ResultidFieldType.GetIdUnionType getId();

      @JsProperty
      void setId(SortTableRequest.ToObjectReturnType0.ResultidFieldType.GetIdUnionType id);

      @JsOverlay
      default void setId(String id) {
        setId(
            Js.<SortTableRequest.ToObjectReturnType0.ResultidFieldType.GetIdUnionType>uncheckedCast(
                id));
      }

      @JsOverlay
      default void setId(Uint8Array id) {
        setId(
            Js.<SortTableRequest.ToObjectReturnType0.ResultidFieldType.GetIdUnionType>uncheckedCast(
                id));
      }
    }

    @JsType(isNative = true, name = "?", namespace = JsPackage.GLOBAL)
    public interface SortsListFieldType {
      @JsOverlay
      static SortTableRequest.ToObjectReturnType0.SortsListFieldType create() {
        return Js.uncheckedCast(JsPropertyMap.of());
      }

      @JsProperty
      String getColumnname();

      @JsProperty
      double getDirection();

      @JsProperty
      boolean isIsabsolute();

      @JsProperty
      void setColumnname(String columnname);

      @JsProperty
      void setDirection(double direction);

      @JsProperty
      void setIsabsolute(boolean isabsolute);
    }

    @JsType(isNative = true, name = "?", namespace = JsPackage.GLOBAL)
    public interface SourceidFieldType {
      @JsOverlay
      static SortTableRequest.ToObjectReturnType0.SourceidFieldType create() {
        return Js.uncheckedCast(JsPropertyMap.of());
      }

      @JsProperty
      double getBatchoffset();

      @JsProperty
      Object getTicket();

      @JsProperty
      void setBatchoffset(double batchoffset);

      @JsProperty
      void setTicket(Object ticket);
    }

    @JsOverlay
    static SortTableRequest.ToObjectReturnType0 create() {
      return Js.uncheckedCast(JsPropertyMap.of());
    }

    @JsProperty
    SortTableRequest.ToObjectReturnType0.ResultidFieldType getResultid();

    @JsProperty
    JsArray<SortTableRequest.ToObjectReturnType0.SortsListFieldType> getSortsList();

    @JsProperty
    SortTableRequest.ToObjectReturnType0.SourceidFieldType getSourceid();

    @JsProperty
    void setResultid(SortTableRequest.ToObjectReturnType0.ResultidFieldType resultid);

    @JsProperty
    void setSortsList(JsArray<SortTableRequest.ToObjectReturnType0.SortsListFieldType> sortsList);

    @JsOverlay
    default void setSortsList(SortTableRequest.ToObjectReturnType0.SortsListFieldType[] sortsList) {
      setSortsList(
          Js.<JsArray<SortTableRequest.ToObjectReturnType0.SortsListFieldType>>uncheckedCast(
              sortsList));
    }

    @JsProperty
    void setSourceid(SortTableRequest.ToObjectReturnType0.SourceidFieldType sourceid);
  }

  public static native SortTableRequest deserializeBinary(Uint8Array bytes);

  public static native SortTableRequest deserializeBinaryFromReader(
      SortTableRequest message, Object reader);

  public static native void serializeBinaryToWriter(SortTableRequest message, Object writer);

  public static native SortTableRequest.ToObjectReturnType toObject(
      boolean includeInstance, SortTableRequest msg);

  public native SortDescriptor addSorts();

  public native SortDescriptor addSorts(SortDescriptor value, double index);

  public native SortDescriptor addSorts(SortDescriptor value);

  public native void clearResultid();

  public native void clearSortsList();

  public native void clearSourceid();

  public native Ticket getResultid();

  public native JsArray<SortDescriptor> getSortsList();

  public native TableReference getSourceid();

  public native boolean hasResultid();

  public native boolean hasSourceid();

  public native Uint8Array serializeBinary();

  public native void setResultid();

  public native void setResultid(Ticket value);

  public native void setSortsList(JsArray<SortDescriptor> value);

  @JsOverlay
  public final void setSortsList(SortDescriptor[] value) {
    setSortsList(Js.<JsArray<SortDescriptor>>uncheckedCast(value));
  }

  public native void setSourceid();

  public native void setSourceid(TableReference value);

  public native SortTableRequest.ToObjectReturnType0 toObject();

  public native SortTableRequest.ToObjectReturnType0 toObject(boolean includeInstance);
}
