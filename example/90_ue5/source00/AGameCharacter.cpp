// no preamble

#include "AGameCharacter.h"
#include "GameCharacter.h"
AGameCharacter::AGameCharacter()
    : Camera{CreateDefaultSubobject<UCameraComponent>(TEXT("Camera"))} {
  // don't perform instance dependent things in constructor. unreal may not call
  // the constructor for each instance

  (PrimaryActorTick.bCanEveryTick) = (true);
}
virtual void AGameCharacter::BeginPlay() { Super::BeginPlay(); }
void AGameCharacter::Tick(float DeltaTime) { Super::Tick(DeltaTime); }
void AGameCharacter::SetupPlayerInputComponent(
    class UInputComponent *PlayerInputComponent) {
  Super::SetupPlayerInputComponent(PlayerInputComponent);
}