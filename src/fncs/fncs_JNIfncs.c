#include "fncs_JNIfncs.h"
#include "fncs.h"  // 假设 libfncs 提供了这个头文件
#include <string.h>
#include <stdlib.h>

JNIEXPORT void JNICALL Java_fncs_JNIfncs_initialize(JNIEnv *env, jclass cls) {
    fncs_initialize();
}

JNIEXPORT void JNICALL Java_fncs_JNIfncs_initialize__Ljava_lang_String_2(JNIEnv *env, jclass cls, jstring config) {
    const char *nativeConfig = (*env)->GetStringUTFChars(env, config, 0);
    fncs_initialize_config(nativeConfig);
    (*env)->ReleaseStringUTFChars(env, config, nativeConfig);
}

JNIEXPORT jboolean JNICALL Java_fncs_JNIfncs_is_1initialized(JNIEnv *env, jclass cls) {
    return fncs_is_initialized() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jlong JNICALL Java_fncs_JNIfncs_time_1request(JNIEnv *env, jclass cls, jlong next) {
    return (jlong)fncs_time_request((fncs_time)next);
}

JNIEXPORT void JNICALL Java_fncs_JNIfncs_publish(JNIEnv *env, jclass cls, jstring key, jstring value) {
    const char *nativeKey = (*env)->GetStringUTFChars(env, key, 0);
    const char *nativeValue = (*env)->GetStringUTFChars(env, value, 0);
    fncs_publish(nativeKey, nativeValue);
    (*env)->ReleaseStringUTFChars(env, key, nativeKey);
    (*env)->ReleaseStringUTFChars(env, value, nativeValue);
}

JNIEXPORT void JNICALL Java_fncs_JNIfncs_publish_1anon(JNIEnv *env, jclass cls, jstring key, jstring value) {
    const char *nativeKey = (*env)->GetStringUTFChars(env, key, 0);
    const char *nativeValue = (*env)->GetStringUTFChars(env, value, 0);
    fncs_publish_anon(nativeKey, nativeValue);
    (*env)->ReleaseStringUTFChars(env, key, nativeKey);
    (*env)->ReleaseStringUTFChars(env, value, nativeValue);
}

JNIEXPORT void JNICALL Java_fncs_JNIfncs_route(JNIEnv *env, jclass cls, jstring from, jstring to, jstring key, jstring value) {
    const char *nativeFrom = (*env)->GetStringUTFChars(env, from, 0);
    const char *nativeTo = (*env)->GetStringUTFChars(env, to, 0);
    const char *nativeKey = (*env)->GetStringUTFChars(env, key, 0);
    const char *nativeValue = (*env)->GetStringUTFChars(env, value, 0);
    fncs_route(nativeFrom, nativeTo, nativeKey, nativeValue);
    (*env)->ReleaseStringUTFChars(env, from, nativeFrom);
    (*env)->ReleaseStringUTFChars(env, to, nativeTo);
    (*env)->ReleaseStringUTFChars(env, key, nativeKey);
    (*env)->ReleaseStringUTFChars(env, value, nativeValue);
}

JNIEXPORT void JNICALL Java_fncs_JNIfncs_die(JNIEnv *env, jclass cls) {
    fncs_die();
}

JNIEXPORT void JNICALL Java_fncs_JNIfncs_end(JNIEnv *env, jclass cls) {
    fncs_finalize();
}

JNIEXPORT void JNICALL Java_fncs_JNIfncs_update_1time_1delta(JNIEnv *env, jclass cls, jlong delta) {
    fncs_update_time_delta((fncs_time)delta);
}

JNIEXPORT jobjectArray JNICALL Java_fncs_JNIfncs_get_1events(JNIEnv *env, jclass cls) {
    size_t size = fncs_get_events_size();
    char **events = fncs_get_events();

    jobjectArray result = (*env)->NewObjectArray(env, size, (*env)->FindClass(env, "java/lang/String"), NULL);
    for (size_t i = 0; i < size; ++i) {
        jstring event = (*env)->NewStringUTF(env, events[i]);
        (*env)->SetObjectArrayElement(env, result, i, event);
    }

    _fncs_free_char_pp(events, size);
    return result;
}

JNIEXPORT jstring JNICALL Java_fncs_JNIfncs_get_1value(JNIEnv *env, jclass cls, jstring key) {
    const char *nativeKey = (*env)->GetStringUTFChars(env, key, 0);
    char *value = fncs_get_value(nativeKey);
    jstring result = (*env)->NewStringUTF(env, value);
    _fncs_free_char_p(value);
    (*env)->ReleaseStringUTFChars(env, key, nativeKey);
    return result;
}

JNIEXPORT jobjectArray JNICALL Java_fncs_JNIfncs_get_1values(JNIEnv *env, jclass cls, jstring key) {
    const char *nativeKey = (*env)->GetStringUTFChars(env, key, 0);
    size_t size = fncs_get_values_size(nativeKey);
    char **values = fncs_get_values(nativeKey);

    jobjectArray result = (*env)->NewObjectArray(env, size, (*env)->FindClass(env, "java/lang/String"), NULL);
    for (size_t i = 0; i < size; ++i) {
        jstring value = (*env)->NewStringUTF(env, values[i]);
        (*env)->SetObjectArrayElement(env, result, i, value);
    }

    _fncs_free_char_pp(values, size);
    (*env)->ReleaseStringUTFChars(env, key, nativeKey);
    return result;
}

JNIEXPORT jobjectArray JNICALL Java_fncs_JNIfncs_get_1keys(JNIEnv *env, jclass cls) {
    size_t size = fncs_get_keys_size();
    char **keys = fncs_get_keys();

    jobjectArray result = (*env)->NewObjectArray(env, size, (*env)->FindClass(env, "java/lang/String"), NULL);
    for (size_t i = 0; i < size; ++i) {
        jstring key = (*env)->NewStringUTF(env, keys[i]);
        (*env)->SetObjectArrayElement(env, result, i, key);
    }

    _fncs_free_char_pp(keys, size);
    return result;
}

JNIEXPORT jstring JNICALL Java_fncs_JNIfncs_get_1name(JNIEnv *env, jclass cls) {
    const char *name = fncs_get_name();
    return (*env)->NewStringUTF(env, name);
}

JNIEXPORT jlong JNICALL Java_fncs_JNIfncs_get_1id(JNIEnv *env, jclass cls) {
    return (jlong)fncs_get_id();
}

JNIEXPORT jlong JNICALL Java_fncs_JNIfncs_get_1simulator_1count(JNIEnv *env, jclass cls) {
    return (jlong)fncs_get_simulator_count();
}

JNIEXPORT jintArray JNICALL Java_fncs_JNIfncs_get_1version(JNIEnv *env, jclass cls) {
    int version[3];
    fncs_get_version(&version[0], &version[1], &version[2]);

    jintArray result = (*env)->NewIntArray(env, 3);
    (*env)->SetIntArrayRegion(env, result, 0, 3, version);
    return result;
}