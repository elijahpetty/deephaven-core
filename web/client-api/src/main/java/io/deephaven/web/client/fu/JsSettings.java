package io.deephaven.web.client.fu;

/**
 * A place to put things like isDevMode() property accessors.
 */
public class JsSettings {

    public static boolean isDevMode() {
        return "true".equals(System.getProperty("dh.dev", "false"));
    }
}
