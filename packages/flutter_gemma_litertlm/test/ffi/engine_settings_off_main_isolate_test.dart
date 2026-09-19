import 'dart:ffi';
import 'dart:isolate';

import 'package:flutter_gemma/core/utils/gemma_log.dart';
import 'package:flutter_gemma_litertlm/src/ffi/litert_lm_bindings.dart';
import 'package:flutter_gemma_litertlm/src/ffi/litert_lm_client.dart';
import 'package:flutter_test/flutter_test.dart';

/// `litert_lm_engine_settings_create` took 81ms on the main isolate of an S25
/// Ultra, just before `initialize` spawned the engine's isolate, and a frame
/// started 98ms late around it. The settings are now made, applied, used and
/// freed on the engine's isolate, reached through [createEngineOffMainIsolate].
///
/// These tests give the client bindings whose every lookup is recorded and
/// throws, and replace the isolate seam, so what they can see is that
/// `initialize` resolves no native symbol at all on the isolate it is called
/// on, and that everything the settings are built from reaches the other one.
///
/// What they cannot see is the timing. That needs a phone.
void main() {
  late Future<int> Function(EngineCreateRequest) originalCreate;
  late GemmaLogLevel originalLogLevel;
  late List<String> mainIsolateLookups;
  late List<EngineCreateRequest> requests;

  LiteRtLmFfiClient clientWithoutNativeLibrary() {
    Pointer<T> lookup<T extends NativeType>(String symbol) {
      mainIsolateLookups.add(symbol);
      throw ArgumentError('no native library on the test host: $symbol');
    }

    return LiteRtLmFfiClient()
      ..bindingsForTest = LiteRtLmBindings.fromLookup(lookup);
  }

  setUp(() {
    originalCreate = createEngineOffMainIsolate;
    originalLogLevel = gemmaLogLevel;
    mainIsolateLookups = [];
    requests = [];
    // Skips the warm-up isolate, which would try to open the real library.
    nativeLibrariesLoadedForTest = true;
  });

  tearDown(() {
    createEngineOffMainIsolate = originalCreate;
    gemmaLogLevel = originalLogLevel;
    nativeLibrariesLoadedForTest = false;
  });

  test('initialize makes the engine settings on the engine isolate, and '
      'calls nothing native on its own', () async {
    createEngineOffMainIsolate = (request) async {
      requests.add(request);
      return 0xE9E;
    };
    final client = clientWithoutNativeLibrary();

    await client.initialize(modelPath: '/models/gemma.litertlm');

    expect(
      mainIsolateLookups,
      isEmpty,
      reason:
          'settings_create, its setters, engine_create and settings_delete all '
          'belong to the engine isolate; a symbol resolved here is a native '
          'call on the UI thread',
    );
    expect(requests, hasLength(1));
    expect(client.isInitialized, isTrue);
  });

  test('the request carries every setting initialize was given', () async {
    createEngineOffMainIsolate = (request) async {
      requests.add(request);
      return 0xE9E;
    };
    gemmaLogLevel = GemmaLogLevel.verbose;

    await clientWithoutNativeLibrary().initialize(
      modelPath: '/models/gemma.litertlm',
      backend: 'cpu',
      maxTokens: 4096,
      cacheDir: '/cache',
      enableVision: true,
      visionBackend: 'gpu',
      maxNumImages: 2,
      audioBackend: 'gpu',
      enableSpeculativeDecoding: false,
    );

    final request = requests.single;
    expect(request.modelPath, '/models/gemma.litertlm');
    expect(request.backend, 'cpu');
    expect(request.maxTokens, 4096);
    expect(request.cacheDir, '/cache');
    expect(request.visionBackend, 'gpu');
    expect(
      request.audioBackend,
      isNull,
      reason: 'audio was not enabled, so no audio encoder is asked for',
    );
    expect(request.maxNumImages, 2);
    expect(request.enableSpeculativeDecoding, isFalse);
    expect(
      request.logLevel,
      GemmaLogLevel.verbose,
      reason: "a spawned isolate starts at the default level, not the caller's",
    );
    expect(
      request.dispatchLibDir,
      isNull,
      reason: 'not an NPU, and the test host is neither Android nor Windows',
    );
    expect(request.disableHwMaskingForNpu, isFalse);
    expect(request.kernelBatchSize, isNull);
  });

  test('a NULL engine fails initialize with the model path, and nothing is '
      'freed on the main isolate', () async {
    createEngineOffMainIsolate = (request) async => 0;
    final client = clientWithoutNativeLibrary();

    await expectLater(
      client.initialize(modelPath: '/models/broken.litertlm'),
      throwsA(
        isA<Exception>().having(
          (e) => e.toString(),
          'message',
          contains(
            'Failed to create engine. Model may be invalid: '
            '/models/broken.litertlm',
          ),
        ),
      ),
    );
    expect(
      mainIsolateLookups,
      isEmpty,
      reason:
          'the settings used to be deleted here after the engine isolate '
          'returned; they are freed on that isolate now, success or not',
    );
    expect(client.isInitialized, isFalse);
  });

  test('the request crosses to a spawned isolate intact', () async {
    EngineCreateRequest requestFor(String modelPath) => EngineCreateRequest(
      modelPath: modelPath,
      backend: 'npu',
      maxTokens: 1024,
      logLevel: GemmaLogLevel.none,
      visionBackend: 'cpu',
      audioBackend: 'gpu',
      cacheDir: '/cache',
      maxNumImages: 3,
      enableSpeculativeDecoding: true,
      dispatchLibDir: '/lib/arm64',
      disableHwMaskingForNpu: true,
      kernelBatchSize: 2,
    );
    final request = requestFor('/models/gemma.litertlm');

    final copy = await Isolate.run(() => request);

    expect(copy.modelPath, request.modelPath);
    expect(copy.backend, request.backend);
    expect(copy.maxTokens, request.maxTokens);
    expect(copy.logLevel, request.logLevel);
    expect(copy.visionBackend, request.visionBackend);
    expect(copy.audioBackend, request.audioBackend);
    expect(copy.cacheDir, request.cacheDir);
    expect(copy.maxNumImages, request.maxNumImages);
    expect(copy.enableSpeculativeDecoding, request.enableSpeculativeDecoding);
    expect(copy.dispatchLibDir, request.dispatchLibDir);
    expect(copy.disableHwMaskingForNpu, request.disableHwMaskingForNpu);
    expect(copy.kernelBatchSize, request.kernelBatchSize);
  });
}
